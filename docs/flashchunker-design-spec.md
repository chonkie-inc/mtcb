# FlashChunker Design Spec

**Goal:** Create a chunker that achieves token-aligned chunks with delimiter awareness at near-Tokie speeds (246 MB/s).

**Target Performance:**
- Quality: Within 1-2pp of RecursiveChunker on retrieval benchmarks
- Speed: ~200 MB/s (20x faster than TokenChunker, leveraging Tokie)

---

## Problem Statement

Current chunkers have trade-offs:

| Chunker | Token-aligned | Delimiter-aware | Speed |
|---------|---------------|-----------------|-------|
| TokenChunker | ✓ | ✗ | ~10 MB/s |
| FastChunker | ✗ | ✓ | ~100 GB/s |
| RecursiveChunker | ✓ | ✓ (hierarchical) | ~10 MB/s |

**Gap:** No chunker combines token alignment + delimiter awareness + high speed.

**New opportunity:** Tokie tokenizer achieves 246 MB/s (55x faster than HuggingFace) with streaming iteration and byte position tracking.

---

## TL;DR - Recommended Approach

Use **Tokie's streaming encoder** to get exact token counts while tracking byte positions, then split at delimiter boundaries:

```
┌────────────────────────────────────────────────────────────┐
│  Text bytes  ──▶  Tokie EncodeIter  ──▶  Token stream      │
│                        │                                    │
│                        ▼                                    │
│              Track cumulative byte position                 │
│              (via O(1) token_len lookup)                    │
│                        │                                    │
│                        ▼                                    │
│              When token_count ≈ target:                     │
│              Find nearest delimiter, emit chunk             │
└────────────────────────────────────────────────────────────┘
```

**Result:** ~200 MB/s with exact token alignment + delimiter awareness.

See "NEW: Tokie-Based Implementation" section for details.

---

## Key Insight

From benchmarks on Gacha with Voyage 3 Large:

- TokenChunker 512T: MRR 0.6403 (token-aligned, no delimiters)
- FastChunker 2048B: MRR 0.6667 (delimiter-aware, byte-based)
- RecursiveChunker 512T: MRR 0.7216 (token-aligned, hierarchical delimiters)

The ~5.5pp gap between FastChunker and RecursiveChunker comes from:
1. **Token alignment** (~2-3pp): Consistent semantic density per chunk
2. **Hierarchical delimiters** (~2-3pp): Preferring paragraph > sentence > word boundaries

FlashChunker targets the token alignment problem while keeping FastChunker's speed.

---

## Design Options

### Option A: Byte Estimation (Simplest)

**Approach:** Convert token target to bytes using average bytes-per-token ratio.

```python
class FlashChunker:
    def __init__(self, chunk_size: int = 512, bytes_per_token: float = 4.0):
        self.chunk_size_tokens = chunk_size
        self.chunk_size_bytes = int(chunk_size * bytes_per_token)

    def chunk(self, text: str) -> List[Chunk]:
        # Use memchunk with estimated byte size
        return fast_chunk(text, size=self.chunk_size_bytes, delimiters="\n.?")
```

**Pros:**
- Trivial implementation
- Full memchunk speed (100+ GB/s)
- No tokenizer dependency at runtime

**Cons:**
- Approximate: 4 bytes/token is average, varies by content
  - English prose: ~4-5 bytes/token
  - Code: ~3-4 bytes/token
  - Unicode-heavy text: ~2-3 bytes/token
- Chunks may vary ±30% from target token count

**Verdict:** Good baseline, but token count variance may hurt retrieval quality.

---

### Option B: Calibrated Byte Estimation

**Approach:** Sample text to calibrate bytes-per-token ratio dynamically.

```python
class FlashChunker:
    def __init__(self, chunk_size: int = 512, tokenizer: str = "auto"):
        self.chunk_size_tokens = chunk_size
        self.tokenizer = load_tokenizer(tokenizer)

    def _calibrate(self, text: str, sample_size: int = 10000) -> float:
        """Sample text to estimate bytes per token."""
        sample = text[:sample_size]
        tokens = self.tokenizer.encode(sample)
        return len(sample.encode("utf-8")) / len(tokens)

    def chunk(self, text: str) -> List[Chunk]:
        bpt = self._calibrate(text)
        byte_size = int(self.chunk_size_tokens * bpt)
        return fast_chunk(text, size=byte_size, delimiters="\n.?")
```

**Pros:**
- Adapts to content type
- One-time tokenization cost (O(sample_size), not O(text_length))
- Still near-memchunk speed for large documents

**Cons:**
- Sample may not represent full document
- Still approximate within document

**Verdict:** Better than Option A, minimal overhead for large documents.

---

### Option C: SentencePiece Metaspace Delimiter

**Approach:** Use the SentencePiece metaspace character (▁) as a delimiter.

Many modern tokenizers (Llama, Mistral, Voyage, etc.) use SentencePiece which prepends ▁ to word-initial tokens. This is a natural token boundary marker already in the text representation.

```python
METASPACE = "▁"  # U+2581, bytes: [0xE2, 0x96, 0x81]

class FlashChunker:
    def __init__(self, chunk_size: int = 512, bytes_per_token: float = 4.0):
        self.chunk_size_bytes = int(chunk_size * bytes_per_token)
        # Use metaspace + standard delimiters
        self.pattern = METASPACE  # or combine with "\n.?"

    def chunk(self, text: str) -> List[Chunk]:
        # memchunk supports multi-byte patterns
        return fast_chunk(text, size=self.chunk_size_bytes, pattern=self.pattern)
```

**Pros:**
- Token boundaries are explicit in text
- Works with SentencePiece-based tokenizers
- Full memchunk speed

**Cons:**
- Only works if text already has metaspace characters (rare)
- Not applicable to tiktoken/BPE tokenizers
- Would require pre-processing text through tokenizer's decode

**Verdict:** Niche use case, not generally applicable.

---

### Option D: Lightweight Token Counting Pass

**Approach:** Do a fast token-counting pass to find approximate split points, then use memchunk to find nearest delimiter.

```python
class FlashChunker:
    def __init__(self, chunk_size: int = 512, tokenizer: str = "auto"):
        self.chunk_size_tokens = chunk_size
        self.tokenizer = load_tokenizer(tokenizer)

    def chunk(self, text: str) -> List[Chunk]:
        # Fast token counting (encode only, no decode)
        tokens = self.tokenizer.encode(text)

        # Find approximate byte positions for token boundaries
        # every chunk_size tokens
        split_token_indices = range(0, len(tokens), self.chunk_size_tokens)

        # Convert token indices to byte positions (approximate)
        # Then use memchunk to find nearest delimiter at each position
        ...
```

**Pros:**
- Exact token alignment at split points
- Delimiter-aware refinement

**Cons:**
- Requires full tokenization pass (defeats speed goal)
- Token-to-byte mapping is non-trivial for most tokenizers

**Verdict:** Not much faster than TokenChunker, defeats the purpose.

---

### Option E: Hybrid Window Approach (Recommended)

**Approach:** Use memchunk for primary chunking, then verify/adjust token counts only at boundaries.

```python
class FlashChunker:
    def __init__(
        self,
        chunk_size: int = 512,  # target tokens
        tokenizer: str = "auto",
        delimiters: str = "\n.?",
        tolerance: float = 0.2,  # allow ±20% variance
    ):
        self.chunk_size_tokens = chunk_size
        self.tokenizer = load_tokenizer(tokenizer)
        self.delimiters = delimiters
        self.tolerance = tolerance

        # Calibrate on first use
        self._bytes_per_token = None

    def _calibrate(self, text: str) -> float:
        """One-time calibration from sample."""
        sample = text[:min(50000, len(text))]
        tokens = self.tokenizer.encode(sample)
        return len(sample.encode("utf-8")) / max(len(tokens), 1)

    def chunk(self, text: str) -> List[Chunk]:
        if self._bytes_per_token is None:
            self._bytes_per_token = self._calibrate(text)

        target_bytes = int(self.chunk_size_tokens * self._bytes_per_token)

        # Primary chunking with memchunk
        raw_chunks = fast_chunk(text, size=target_bytes, delimiters=self.delimiters)

        # Optional: verify token counts, merge/split if outside tolerance
        # This is O(n) but with very small constant factor
        final_chunks = self._verify_and_adjust(raw_chunks)

        return final_chunks

    def _verify_and_adjust(self, chunks: List[Chunk]) -> List[Chunk]:
        """Verify chunks are within tolerance, adjust if needed."""
        min_tokens = int(self.chunk_size_tokens * (1 - self.tolerance))
        max_tokens = int(self.chunk_size_tokens * (1 + self.tolerance))

        result = []
        buffer = ""

        for chunk in chunks:
            candidate = buffer + chunk.text
            token_count = self.tokenizer.count_tokens(candidate)

            if token_count < min_tokens:
                # Too small, buffer for merging
                buffer = candidate
            elif token_count > max_tokens:
                # Too large, need to split
                # Use binary search to find split point
                result.extend(self._split_oversized(candidate, max_tokens))
                buffer = ""
            else:
                # Within tolerance
                result.append(Chunk(text=candidate, ...))
                buffer = ""

        if buffer:
            result.append(Chunk(text=buffer, ...))

        return result
```

**Pros:**
- Token count verification ensures quality
- memchunk does 99% of the work
- Only edge cases trigger additional processing
- Adaptive calibration per document

**Cons:**
- Verification pass adds ~10% overhead
- More complex implementation
- Still requires tokenizer at runtime

**Verdict:** Best balance of speed and quality.

---

---

## NEW: Tokie-Based Implementation (Recommended)

After exploring the Tokie codebase, we found it already provides the primitives needed for fast token-aligned chunking.

### What Tokie Provides

1. **Streaming encoder iterator** (`EncodeIter`): Yields tokens lazily with internal byte position tracking
2. **O(1) token length lookup** (`token_len()`): Get byte length for any token instantly
3. **memchunk integration**: Already uses memchunk for parallel chunking at whitespace
4. **246 MB/s throughput**: 55x faster than HuggingFace tokenizers

### Key Data Structures

```rust
// From tokie/src/encoder/backtracking.rs
pub struct EncodeIter<'a> {
    encoder: &'a BacktrackingBytePairEncoder,
    text: &'a [u8],
    pos: usize,        // Current byte position - THIS IS KEY
    buffer: VecDeque<TokenId>,
    // ...
}

// From tokie/src/decoder.rs
impl Decoder {
    // O(1) lookup - flat buffer with offset array
    pub fn token_len(&self, token: TokenId) -> usize {
        let start = self.offsets[token as usize];
        let end = self.offsets[token as usize + 1];
        (end - start) as usize
    }
}
```

### FlashChunker Algorithm (Rust)

```rust
use tokie::Tokenizer;
use memchunk::chunk_offsets;

pub struct FlashChunker {
    tokenizer: Tokenizer,
    chunk_size: usize,      // target tokens
    delimiters: Vec<u8>,    // e.g., b"\n.?"
}

impl FlashChunker {
    pub fn chunk(&self, text: &str) -> Vec<Chunk> {
        let bytes = text.as_bytes();
        let decoder = self.tokenizer.decoder();

        let mut chunks = Vec::new();
        let mut chunk_start = 0;
        let mut token_count = 0;
        let mut byte_pos = 0;

        // Stream through tokens
        for token in self.tokenizer.encode_iter(text) {
            let token_bytes = decoder.token_len(token);
            token_count += 1;
            byte_pos += token_bytes;

            // Check if we've reached target chunk size
            if token_count >= self.chunk_size {
                // Find nearest delimiter AFTER current position
                let search_window = &bytes[byte_pos..];
                let delimiter_offset = self.find_nearest_delimiter(search_window);

                // Create chunk
                let chunk_end = byte_pos + delimiter_offset;
                chunks.push(Chunk {
                    text: String::from_utf8_lossy(&bytes[chunk_start..chunk_end]).into(),
                    start_index: chunk_start,
                    end_index: chunk_end,
                    token_count,
                });

                // Reset for next chunk
                chunk_start = chunk_end;
                token_count = 0;
                byte_pos = chunk_end;
            }
        }

        // Handle remaining text
        if chunk_start < bytes.len() {
            chunks.push(Chunk {
                text: String::from_utf8_lossy(&bytes[chunk_start..]).into(),
                start_index: chunk_start,
                end_index: bytes.len(),
                token_count,
            });
        }

        chunks
    }

    fn find_nearest_delimiter(&self, window: &[u8]) -> usize {
        // Use memchunk's SIMD-accelerated delimiter search
        // Search forward for nearest delimiter
        for (i, &byte) in window.iter().enumerate() {
            if self.delimiters.contains(&byte) {
                return i + 1; // Include delimiter in current chunk
            }
        }
        0 // No delimiter found, split here
    }
}
```

### Optimized Version with Lookahead Window

```rust
impl FlashChunker {
    /// Chunk with delimiter-aware boundaries
    ///
    /// Strategy: When approaching chunk_size, look ahead for the nearest
    /// delimiter within a tolerance window (±10% of chunk_size).
    pub fn chunk_optimized(&self, text: &str) -> Vec<Chunk> {
        let bytes = text.as_bytes();
        let decoder = self.tokenizer.decoder();

        let min_tokens = (self.chunk_size as f32 * 0.9) as usize;
        let max_tokens = (self.chunk_size as f32 * 1.1) as usize;

        let mut chunks = Vec::new();
        let mut chunk_start_byte = 0;
        let mut chunk_tokens = Vec::new();
        let mut byte_pos = 0;

        let mut iter = self.tokenizer.encode_iter(text).peekable();

        while let Some(token) = iter.next() {
            let token_len = decoder.token_len(token);
            chunk_tokens.push(token);
            byte_pos += token_len;

            // In the sweet spot - look for delimiter
            if chunk_tokens.len() >= min_tokens {
                // Check if we're at a delimiter
                if byte_pos < bytes.len() {
                    let next_byte = bytes[byte_pos];
                    if self.delimiters.contains(&next_byte) || chunk_tokens.len() >= max_tokens {
                        // Good split point - emit chunk
                        let chunk_end = byte_pos;
                        chunks.push(Chunk {
                            text: String::from_utf8_lossy(&bytes[chunk_start_byte..chunk_end]).into(),
                            start_index: chunk_start_byte,
                            end_index: chunk_end,
                            token_count: chunk_tokens.len(),
                        });

                        chunk_start_byte = chunk_end;
                        chunk_tokens.clear();
                    }
                }
            }
        }

        // Remaining chunk
        if !chunk_tokens.is_empty() {
            chunks.push(Chunk {
                text: String::from_utf8_lossy(&bytes[chunk_start_byte..]).into(),
                start_index: chunk_start_byte,
                end_index: bytes.len(),
                token_count: chunk_tokens.len(),
            });
        }

        chunks
    }
}
```

### Performance Analysis

| Operation | Time Complexity | Throughput |
|-----------|-----------------|------------|
| Token streaming | O(n) | 246 MB/s |
| token_len lookup | O(1) | Negligible |
| Delimiter check | O(1) per byte | Negligible |
| **Total** | O(n) | **~200 MB/s** |

**Comparison:**
- Tokie-based FlashChunker: ~200 MB/s
- TokenChunker (HuggingFace): ~10 MB/s
- Speedup: **20x**

### Why This Works

1. **No double tokenization**: TokenChunker encodes, chunks by count, then decodes. FlashChunker streams once.

2. **Byte position is free**: `EncodeIter` internally tracks `pos`, and `token_len()` is O(1). No byte↔token mapping needed.

3. **Delimiter check is cheap**: Single byte comparison at chunk boundaries, not full text scan.

4. **Parallel-friendly**: For very large texts, can use memchunk to split at whitespace first, then run FlashChunker on each segment.

### Integration Path

**Phase 1: Rust Implementation**
1. Create new `flashchunk` crate (depends on `tokie` + `memchunk`)
2. Implement `FlashChunker` struct with `chunk()` method
3. Return `Vec<ChunkInfo>` with `(start, end, token_count)`

**Phase 2: Python Bindings (PyO3)**
```python
from flashchunk import FlashChunker

chunker = FlashChunker(
    tokenizer="gpt2",  # or path to tokenizer.json
    chunk_size=512,
    delimiters="\n.?",
)

chunks = chunker.chunk("Long text here...")
# Returns: [Chunk(text="...", start=0, end=1024, tokens=512), ...]
```

**Phase 3: Chonkie Integration**
```python
from chonkie import FlashChunker

chunker = FlashChunker(
    chunk_size=512,
    tokenizer="voyage-3-large",
)
```

---

## Original Design Options (Pre-Tokie Analysis)

The following options were considered before discovering Tokie's capabilities. They are preserved for reference but **Option E with Tokie integration is now recommended**.

---

## DEPRECATED: Recommended Implementation: Option E (Hybrid)

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FlashChunker                         │
├─────────────────────────────────────────────────────────┤
│  1. Calibration Layer                                   │
│     - Sample text to estimate bytes_per_token           │
│     - Cache ratio for subsequent calls                  │
├─────────────────────────────────────────────────────────┤
│  2. Primary Chunking (memchunk)                         │
│     - Convert token target to byte target               │
│     - Use memchunk with delimiter awareness             │
│     - Returns approximate chunks at ~100 GB/s           │
├─────────────────────────────────────────────────────────┤
│  3. Verification Layer (optional)                       │
│     - Count tokens in each chunk                        │
│     - Merge undersized chunks                           │
│     - Split oversized chunks at delimiters              │
│     - Ensures chunks within tolerance                   │
└─────────────────────────────────────────────────────────┘
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `chunk_size` | int | 512 | Target tokens per chunk |
| `chunk_overlap` | int | 0 | Token overlap between chunks |
| `tokenizer` | str | "auto" | Tokenizer for calibration/verification |
| `delimiters` | str | "\n.?" | Delimiters for boundary detection |
| `tolerance` | float | 0.2 | Allowed variance from target (±20%) |
| `verify` | bool | True | Enable verification pass |

### Modes

1. **Fast Mode** (`verify=False`):
   - Pure memchunk with calibrated byte estimation
   - Speed: ~100 GB/s
   - Token variance: ±30%

2. **Balanced Mode** (`verify=True`, default):
   - memchunk + verification pass
   - Speed: ~10 GB/s (still 1000x faster than TokenChunker)
   - Token variance: within tolerance (±20%)

3. **Strict Mode** (`tolerance=0.05`):
   - Tighter tolerance, more adjustments
   - Speed: ~1 GB/s
   - Token variance: ±5%

---

## Implementation Plan

### Phase 1: Core Implementation

1. **FlashChunker class** with calibration + memchunk integration
2. **Tokenizer integration** using Chonkie's existing tokenizer module
3. **Basic verification** (count tokens, flag out-of-tolerance chunks)

### Phase 2: Optimization

1. **Merge/split logic** for out-of-tolerance chunks
2. **Overlap handling** for chunk continuity
3. **Batch processing** for multiple documents

### Phase 3: Benchmarking

1. **Speed benchmarks** vs TokenChunker, FastChunker
2. **Quality benchmarks** on Gacha (MRR, HIT@k)
3. **Tolerance tuning** to find optimal speed/quality trade-off

---

## Expected Results (Updated with Tokie)

Based on the analysis with Tokie integration:

| Chunker | Speed | MRR (predicted) |
|---------|-------|-----------------|
| TokenChunker | ~10 MB/s | 0.64 |
| FastChunker | ~100 GB/s | 0.67 |
| **FlashChunker (Tokie)** | **~200 MB/s** | **0.70-0.72** |
| RecursiveChunker | ~10 MB/s | 0.72 |

**Key insight:** With exact token counts (not estimates) + delimiter awareness, FlashChunker should match or approach RecursiveChunker quality while being **20x faster**.

The only remaining gap is hierarchical delimiter preference (prefer `\n\n` over `\n` over `.`). This could be added as an enhancement.

---

## Open Questions

1. **Is hierarchical delimiter preference worth adding?**
   - Could add `delimiter_priority` parameter: `[b"\n\n", b"\n", b".", b"?"]`
   - Would need backward search within tolerance window
   - May close the remaining 1-2pp gap to RecursiveChunker

2. **Should we support chunk overlap?**
   - Current design produces non-overlapping chunks
   - Overlap would require tracking token history

3. **Where should this live?**
   - New `swiftchunk` crate depending on `tokie` and `memchunk`?
   - Integrated into `chonkie` directly?
   - As a feature flag in `tokie` itself?

4. **Python binding strategy?**
   - PyO3 direct bindings (fastest)
   - Or wrap via Chonkie's existing Python interface

---

## Required Changes to Tokie

### Minimal Change (Recommended)

**One-line addition to expose byte position from `EncodeIter`:**

```rust
// In tokie/src/encoder/backtracking.rs
impl<'a> EncodeIter<'a> {
    /// Get current byte position in the text
    pub fn byte_pos(&self) -> usize {
        self.pos
    }
}
```

This exposes the already-tracked `pos` field, allowing FlashChunker to get byte position without computing it externally.

### Workaround (No Tokie Changes)

If we don't want to modify Tokie, we can compute byte position externally:

```rust
let decoder = tokenizer.decoder();
let mut byte_pos = 0;

for token in tokenizer.encode_iter(text) {
    byte_pos += decoder.token_len(token);  // O(1) lookup
    // ... chunking logic
}
```

This works but adds a function call per token. The overhead is minimal since `token_len()` is O(1), but exposing `pos` directly is cleaner.

### Streaming Support by Encoder Type

`encode_iter()` is available for all encoder types, but **only Backtracking has true streaming**:

```rust
// From tokie/src/encoder/mod.rs
pub fn encode_iter<'a>(&'a self, text: &'a [u8]) -> EncoderIter<'a> {
    match self {
        Encoder::Backtracking(e) => EncoderIter::Backtracking(e.encode_iter(text)),
        // Others collect all tokens first, then iterate
        Encoder::Simple(_) | Encoder::WordPiece(_) | ... => {
            EncoderIter::Collected(self.encode(text).into_iter())
        }
    }
}
```

| Encoder | True Streaming | Models |
|---------|----------------|--------|
| Backtracking | ✅ Yes | Voyage, OpenAI, Llama 3, Qwen, ModernBERT |
| Simple | ❌ Collects first | Mistral Nemo, RoBERTa, DeepSeek |
| WordPiece | ❌ Collects first | BERT, BGE, E5, GTE |
| SentencePiece | ❌ Collects first | Llama 1/2, Mistral 7B, Gemma |
| Unigram | ❌ Collects first | T5, XLM-RoBERTa, ALBERT |

**Implication for FlashChunker:**
- **Voyage, OpenAI, Llama 3, Qwen:** Full streaming benefit (~200 MB/s)
- **Other encoders:** Still works, but tokenizes full text upfront (no streaming benefit)

**Good news:** Voyage uses Backtracking, so FlashChunker gets true streaming for the primary use case (RAG with Voyage embeddings).

### Deep Dive: Why Some Encoders Can't Stream

**Backtracking (CAN stream):**
- Left-to-right greedy matching with bounded backtrack buffer (8 tokens)
- Tokens are confirmed and emitted as the buffer fills
- `EncodeIter` maintains `pos` (byte position) as it yields tokens

**WordPiece (COULD stream - not yet implemented):**
```rust
// The encode loop is already left-to-right
loop {
    while pos < word.len() {
        if let Some(next_state) = self.try_transition(state, word[pos]) {
            state = next_state;
            pos += 1;
            if let Some(output) = self.matcher.outputs(state).next() {
                last_match = Some((pos, output.pattern_id));
            }
        } else {
            // Emit last_match here - natural streaming point
            result.push(token_id);
            state = anchor;
        }
    }
}
```
- Greedy longest-match via DAAC traversal
- Tokens can be emitted as soon as a match is confirmed
- **Adding streaming would be medium effort**

**Simple BPE (CANNOT stream):**
```rust
// Merge loop modifies tokens in-place - any token might get merged later
while len > 1 {
    // Find lowest-rank merge across ALL pairs
    for i in 0..len - 1 {
        if let Some(&(merged, rank)) = self.pair_lookup.get(...) {
            if rank < best_rank { best_pos = i; }
        }
    }
    // Apply merge - affects tokens we thought were "done"
    tokens[best_pos] = best_merged;
    tokens.copy_within(best_pos + 2..len, best_pos + 1);
    len -= 1;
}
```
- O(n²) algorithm: repeatedly find and apply lowest-rank merge
- A token at position 0 might get merged with position 1 on the last iteration
- **Cannot emit any token until all merges complete**

**SentencePiece BPE (CANNOT stream):**
```rust
// Radix heap processes merges globally by rank
while let Some(entry) = heap.pop() {
    // Validate entry is still current (symbols might have been merged)
    if left.len == 0 || right.len == 0 { continue; }
    if left.next != entry.right { continue; }

    // Perform merge
    symbols[left_idx].token = merged_token;
    symbols[right_idx].len = 0;  // Mark as merged away
}
```
- Uses radix heap to process merges in rank order
- Same fundamental issue: any symbol might get merged later
- **Cannot emit until heap is exhausted**

**Unigram (CANNOT stream):**
```rust
// Forward pass: compute best score to reach each position
for pos in 0..n {
    for &(end, token_id) in &matches_at[pos] {
        let new_score = current_score + token_score;
        if new_score > best_score[end] {
            best_score[end] = new_score;
            backptr[end] = (token_id, pos);
        }
    }
}

// Backward pass: reconstruct optimal path (REQUIRES COMPLETE FORWARD PASS)
let mut pos = n;
while pos > 0 {
    let (token_id, start_pos) = backptr[pos];
    tokens.push(token_id);
    pos = start_pos;
}
tokens.reverse();
```
- Viterbi dynamic programming: forward pass + backward pass
- The optimal token at position 0 depends on scores computed at position n
- **Cannot emit any token until backward pass completes**

### Workaround: Chunk-Based Pseudo-Streaming

For non-streaming encoders, SentencePiece already implements a workaround:

```rust
// From sentencepiece.rs - split at metaspace boundaries, encode each chunk
pub fn encode_chunked(&self, text: &[u8], state: &mut EncodeState, chunk_size: usize) -> Vec<TokenId> {
    for chunk_bytes in chunk(text)
        .size(chunk_size)
        .pattern(&METASPACE)
        .prefix()
        .consecutive()
        .forward_fallback()
    {
        let chunk_tokens = self.encode_with_state(chunk_bytes, state);
        result.extend_from_slice(chunk_tokens);
    }
}
```

This provides:
- ✅ Bounded memory usage (process one chunk at a time)
- ✅ Cache-friendly (chunk fits in L1/L2)
- ❌ Not true token-by-token streaming
- ❌ Chunk boundaries must be at safe positions (whitespace/metaspace)

### Summary: FlashChunker Compatibility

| Encoder | Streaming Mode | FlashChunker Benefit |
|---------|---------------|---------------------|
| Backtracking | True streaming | Full (~200 MB/s) |
| WordPiece | Could add streaming | Full (with Tokie enhancement) |
| Simple | Chunk-based only | Partial (memory bounded) |
| SentencePiece | Chunk-based only | Partial (memory bounded) |
| Unigram | Chunk-based only | Partial (memory bounded) |

**Primary use case (Voyage embeddings):** Uses Backtracking → Full streaming support ✅

### Optional Enhancements

These are NOT required but could be added later:

1. **Expose underlying text reference:**
   ```rust
   impl<'a> EncodeIter<'a> {
       pub fn text(&self) -> &'a [u8] {
           self.text
       }
   }
   ```

2. **Add true streaming to other encoder types:**
   Currently only `BacktrackingBytePairEncoder` has true streaming. Adding streaming to `Simple`, `WordPiece`, `SentencePiece`, and `Unigram` would broaden FlashChunker's streaming benefit to more models.

3. **Delimiter search utility:**
   ```rust
   // Could live in tokie or flashchunk
   pub fn find_delimiters(text: &[u8], delimiters: &[u8]) -> Vec<usize>
   ```

### Crate Structure

```
flashchunk/
├── Cargo.toml          # depends on tokie, memchunk
├── src/
│   ├── lib.rs          # FlashChunker implementation
│   ├── chunk.rs        # Chunk struct and helpers
│   └── python.rs       # PyO3 bindings (optional feature)
```

**Cargo.toml:**
```toml
[package]
name = "flashchunk"
version = "0.1.0"

[dependencies]
tokie = "0.2"
memchunk = "0.1"

[features]
default = []
python = ["pyo3"]

[dependencies.pyo3]
version = "0.20"
optional = true
```

### Summary

| Change | Location | Required? | Effort |
|--------|----------|-----------|--------|
| `byte_pos()` getter | tokie | Recommended | 1 line |
| External byte tracking | flashchunk | Workaround | 3 lines |
| `text()` getter | tokie | Optional | 1 line |
| Streaming for other encoders | tokie | Optional | Medium |
| Delimiter search util | flashchunk | Optional | ~20 lines |

**Bottom line:** FlashChunker can be built with **zero changes to Tokie** using the external byte tracking workaround. Adding the `byte_pos()` getter is a nice-to-have for cleaner code.

---

## Appendix: Bytes-per-Token Reference

| Tokenizer | Content Type | Avg Bytes/Token |
|-----------|--------------|-----------------|
| cl100k_base (OpenAI) | English prose | 4.0-4.5 |
| cl100k_base (OpenAI) | Code | 3.0-3.5 |
| voyage-3-large | English prose | 4.0-4.5 |
| llama-3 | English prose | 3.5-4.0 |
| General | Unicode-heavy | 2.0-3.0 |
| General | ASCII-only | 4.0-5.0 |

Default of 4.0 bytes/token is reasonable for English text with most tokenizers.
