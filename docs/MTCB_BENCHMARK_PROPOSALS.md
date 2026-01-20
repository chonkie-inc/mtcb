# MTCB: Massive Text Chunking Benchmark

## Vision

MTCB aims to be the definitive benchmark suite for evaluating text chunking algorithms, analogous to what MTEB is for embeddings. By providing standardized evaluation across diverse document types and domains, MTCB will enable:

- Objective comparison of chunking strategies
- Domain-specific chunker selection guidance
- Progress tracking for the chunking research community

---

## Current Benchmarks

| Benchmark | Domain | Documents | What It Tests |
|-----------|--------|-----------|---------------|
| **Gacha** | Literary/Textbooks | 100 Gutenberg books | Long-form prose chunking, NIAH-style retrieval |
| **Macha** | Technical Docs | 957 GitHub READMEs | Semi-structured markdown, developer Q&A |

---

## Proposed New Benchmarks

### 1. Code Chunking Benchmark

**Working Name**: `Cocha` (Code + cha) or `Gicha` (Git + cha)

> See [Naming Discussion](#naming-convention) for alternatives

**Source Data**:
- Top 500-1000 GitHub repositories by stars
- Focus on Python, JavaScript, TypeScript, Go, Rust (popular + diverse syntax)
- Include: source files, not just READMEs

**What It Tests**:
- Function/class boundary preservation
- Import statement grouping
- Docstring-to-code association
- Multi-file context (cross-file references)

**Question Types**:
- "What does the `process_data` function do?"
- "Where is the `UserModel` class defined?"
- "What parameters does `authenticate()` accept?"
- "What does this file import from external packages?"

**Why It Matters**:
- Code chunking is fundamentally different from prose—syntax matters
- `CodeChunker` exists in Chonkie but has no benchmark
- Code RAG (Copilot-style assistants, codebase Q&A) is a massive use case

**Chunk-Must-Contain Examples**:
- Full function definition for function questions
- Class definition + constructor for class questions
- Import block for dependency questions

---

### 2. Mocha (Academic/Scientific Papers)

**Source Data**:
- ArXiv open-access papers (CS, Physics, Math sections)
- PubMed Central open-access biomedical papers
- ~500-1000 papers across domains

**What It Tests**:
- Section boundary handling (Abstract, Methods, Results, Discussion)
- Citation preservation (references kept with claims)
- Equation and figure reference coherence
- Dense technical vocabulary

**Question Types**:
- "What method did the authors use for X?"
- "What were the main results of the experiment?"
- "Which paper introduced technique Y?" (citation retrieval)
- "What dataset was used for evaluation?"

**Why It Matters**:
- Research RAG is exploding (Elicit, Consensus, ScholarAI)
- Papers have unique structure: abstract, citations, figures, equations
- Information density is extremely high

**Chunk-Must-Contain Examples**:
- Methods section paragraphs for methodology questions
- Results tables/figures captions for results questions
- Bibliography entries for citation questions

---

### 3. Sencha (Structured/Tabular Documents)

**Source Data**:
- Wikipedia tables (infoboxes, comparison tables)
- Government open data (Census, economic reports)
- Sports statistics, leaderboards
- ~1000+ documents with embedded tables

**What It Tests**:
- Table boundary preservation (don't split mid-table)
- Row/column header context retention
- Table-to-prose transition handling
- Numeric precision in retrieval

**Question Types**:
- "What was the GDP of France in 2023?"
- "Which country has the highest life expectancy?"
- "Compare the specs of Product A vs Product B"
- "What are the nutritional facts for X?"

**Why It Matters**:
- Tables are notoriously hard to chunk—splitting ruins context
- `TableChunker` exists but has no benchmark
- Enterprise data is heavily tabular (Excel, databases, reports)

**Chunk-Must-Contain Examples**:
- Complete table (or complete row with headers) for table questions
- Surrounding context paragraph for table interpretation questions

---

### 4. Hojicha (Legal/Contract Documents)

**Source Data**:
- Court opinions (CourtListener, Case.law open access)
- Terms of Service / Privacy Policies (TOSDR dataset)
- Open contracts (CUAD dataset)
- ~500-1000 documents

**What It Tests**:
- Long, complex sentence handling
- Cross-reference preservation ("as defined in Section 3.2")
- Clause boundary detection
- Defined terms and their definitions

**Question Types**:
- "What are the termination conditions?"
- "What does 'Confidential Information' mean in this contract?"
- "What is the liability cap?"
- "Under what circumstances can the agreement be modified?"

**Why It Matters**:
- Legal RAG is a killer enterprise use case
- Legal text has unique structure: numbered sections, defined terms, cross-refs
- Chunking errors in legal contexts have real consequences

**Chunk-Must-Contain Examples**:
- Full clause for clause-specific questions
- Definition paragraph for "what does X mean" questions
- Multiple related sections for cross-reference questions

---

### 5. Chacha (Conversational/Dialogue)

**Source Data**:
- Ubuntu Dialogue Corpus (tech support)
- Reddit comment threads (pushshift archives)
- Meeting transcripts (AMI Corpus, ICSI)
- Customer support logs (if available open-source)
- ~1000+ conversations

**What It Tests**:
- Speaker attribution preservation
- Turn boundary handling
- Context window spanning multiple turns
- Topic shift detection within conversations

**Question Types**:
- "What solution did the support agent suggest?"
- "What was the user's original problem?"
- "Did anyone disagree with the proposal?"
- "What was the final decision in the meeting?"

**Why It Matters**:
- Chat-based RAG (support bots, meeting summarizers) is growing
- Dialogue chunking must preserve who-said-what
- Multi-turn context is fundamentally different from documents

**Chunk-Must-Contain Examples**:
- Speaker turn(s) containing the answer
- Full exchange (question + answer turns) for Q&A retrieval

---

### 6. Ficha (Financial Documents)

**Source Data**:
- SEC 10-K and 10-Q filings (EDGAR database)
- Earnings call transcripts (open sources)
- Annual reports (if openly available)
- ~500-1000 documents

**What It Tests**:
- Numeric precision and context
- Table + prose interleaving
- Temporal references ("Q3 2024", "year-over-year")
- Forward-looking statement handling

**Question Types**:
- "What was the company's revenue in Q3?"
- "What are the main risk factors mentioned?"
- "How did gross margin change year-over-year?"
- "What is management's outlook for next quarter?"

**Why It Matters**:
- Financial RAG is high-value (Bloomberg, FactSet, analysts)
- Mix of narrative + numbers + tables is challenging
- Temporal context is critical (which quarter? which year?)

**Chunk-Must-Contain Examples**:
- Financial table or paragraph with specific numbers
- Risk factor section for risk questions
- MD&A paragraphs for outlook questions

---

## Priority Ranking

| Priority | Benchmark | Rationale |
|----------|-----------|-----------|
| **P0** | Code (Cocha/Gicha) | Huge use case, distinct chunking needs, no existing benchmark |
| **P0** | Academic (Mocha) | Research RAG is booming, unique document structure |
| **P1** | Tables (Sencha) | Known pain point, high practical impact |
| **P1** | Legal (Hojicha) | Enterprise killer app, complex linguistic patterns |
| **P2** | Dialogue (Chacha) | Underexplored, growing use case |
| **P2** | Financial (Ficha) | High value but overlaps with tables + legal |

---

## Naming Convention

Current names follow a pattern ending in `-cha` (茶 = tea in Japanese):

| Benchmark | Name Origin | Notes |
|-----------|-------------|-------|
| Gacha | ガチャ (gacha games) | Random draw / capsule toy machines |
| Macha | 抹茶 (matcha) | Green tea powder |

**Proposed names follow tea theme**:

| Benchmark | Proposed Name | Tea Reference | Alternatives |
|-----------|---------------|---------------|--------------|
| Code | **Cocha** | Phonetic (code+cha) | Gicha (git+cha), Socha (source+cha), Pycha (python+cha) |
| Academic | **Mocha** | Coffee+chocolate drink | Rocha (research+cha) |
| Tables | **Sencha** | 煎茶 (Japanese green tea) | Tacha (table+cha) |
| Legal | **Hojicha** | ほうじ茶 (roasted tea) | Lecha (legal+cha) |
| Dialogue | **Chacha** | Playful doubling | Talkcha, Convcha |
| Financial | **Ficha** | Phonetic (finance+cha) | Fincha, Moncha |

### Why "Cocha" for Code?

Honestly, it's a weak connection—just phonetic similarity. Better alternatives:

1. **Gicha** - Git + cha (most code is in git repos)
2. **Socha** - Source + cha (source code)
3. **Repacha** - Repo + cha (repositories)
4. **Syncha** - Syntax + cha (emphasizes syntactic chunking)

**Recommendation**: Use **Gicha** or **Syncha** for stronger semantic connection.

---

## Cross-Cutting Evaluation Dimensions

Beyond per-dataset metrics (Recall@k, MRR@k), consider adding:

### 1. Boundary Quality Score
- Measures whether chunk boundaries fall at semantic breaks
- Could use perplexity delta at boundaries as a proxy

### 2. Information Density Variance
- Are chunks evenly informative or wildly unbalanced?
- High variance = some chunks useless, others overloaded

### 3. Cross-Reference Preservation Rate
- For documents with internal references, what % are kept with their referent?

### 4. Chunking Throughput
- Tokens/second for each chunker
- Important for production systems

---

## Dataset Creation Methodology

> **Reference Implementation**: See `/Users/bhavnick/Projects/datasets/ficha/scripts/` for the complete Ficha pipeline that serves as the template for all MTCB datasets.

### General Pipeline

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  1. Download    │────▶│  2. Convert      │────▶│  3. Build        │────▶│  4. Generate    │
│  Source Data    │     │  to Markdown     │     │  Corpus          │     │  Questions      │
└─────────────────┘     └──────────────────┘     └──────────────────┘     └─────────────────┘
                                                                                   │
                                                                                   ▼
                                                                          ┌─────────────────┐
                                                                          │  5. Upload to   │
                                                                          │  HuggingFace    │
                                                                          └─────────────────┘
```

---

### Step 1: Source Data Collection

| Benchmark | Source | Collection Method |
|-----------|--------|-------------------|
| Ficha (reference) | SEC EDGAR | `download_f100.py` - Fortune 100 10-Ks |
| Cocha (Code) | GitHub API | Top repos by stars, filter by language |
| Mocha (Academic) | ArXiv API / PubMed | Bulk download, filter by license |
| Sencha (Tables) | Wikipedia dumps | Extract pages with tables |
| Hojicha (Legal) | EDGAR / CourtListener | API access, public domain |
| Chacha (Dialogue) | Ubuntu Corpus / Reddit | Existing datasets, pushshift |

**Key considerations from Ficha:**
- Proper rate limiting (SEC requires 0.1-0.15s delay between requests)
- Required User-Agent headers for API compliance
- Curated source list (Fortune 100) vs. arbitrary scraping

---

### Step 2: Convert to Clean Text (Markdown)

**Reference**: `html_to_markdown.py`

For HTML sources, conversion involves:

```python
# 1. Clean HTML with BeautifulSoup
- Remove: scripts, styles, meta tags, noscript
- Remove: hidden elements (display:none, visibility:hidden)
- Remove: empty layout elements (spans, divs, p with no content)

# 2. Convert with html2text
h = html2text.HTML2Text()
h.body_width = 0        # No line wrapping
h.ignore_images = True  # Skip images
h.pad_tables = True     # Preserve table structure

# 3. Post-process
- Normalize whitespace (max 3 consecutive newlines)
- Clean table rows (merge currency symbols with values)
- Remove orphaned table separators
```

**Critical table handling** (from Ficha):
```python
# Merge currency symbols with following values
# ['$', '416,161'] -> ['$416,161']
if cell in ('$', '€', '£', '¥') and next_cell_is_number:
    merged.append(cell + next_cell)

# Merge trailing % with previous number
# ['45.2', '%'] -> ['45.2%']
if cell == '%' and previous_cell_is_number:
    merged[-1] = merged[-1] + '%'
```

---

### Step 3: Build Corpus Dataset

**Reference**: `build_corpus.py`

Structure each document with metadata:

```python
corpus_entry = {
    "ticker": "AAPL",                    # Unique identifier
    "company": "Apple Inc.",             # Human-readable name
    "filing_type": "10-K",               # Document type
    "filing_date": "2024-10-31",         # Date
    "text": "..."                        # Full document text
}
```

Upload as HuggingFace dataset:
```python
dataset = Dataset.from_list(corpus)
dataset.push_to_hub(
    "chonkie-ai/ficha",
    config_name="corpus",    # Separate config for corpus vs questions
    split="train"
)
```

---

### Step 4: Question Generation (The Critical Step)

**Reference**: `generate_questions.py`

#### 4a. Chunking Strategy
```python
CHUNK_SIZE = 8192   # tokens
OVERLAP = 512       # token overlap between chunks

# Use tiktoken for consistent tokenization
tokenizer = tiktoken.get_encoding("cl100k_base")
```

#### 4b. Question Generation Prompt

The prompt is highly specific and structured:

```python
QGEN_TEMPLATE = """<role>
You are an expert AI assistant tasked with generating high-quality
question-answer pairs from {document_type} excerpts.
</role>

<instructions>
1. **Understand the Passage**: Identify key information - figures,
   descriptions, dates, percentages, specific details.

2. **Formulate a Question**:
   - MUST be answerable solely from the passage (no external knowledge)
   - Target specific information, not broad summaries
   - Prefer: What, When, Where, Which, How questions
   - Focus on factual, concise answers

3. **Provide a Concise Answer**:
   - Direct & self-contained (makes sense without the question)
   - 1-2 sentences max
   - Derived verbatim or close paraphrase from passage

4. **Cite the Supporting Passage**:
   - EXACT, CONTIGUOUS quote from the passage
   - No ellipses or additions
   - Minimal sufficient context (shortest block that fully supports answer)

5. **Avoid**:
   - Questions requiring synthesis from non-adjacent parts
   - Yes/No questions
   - Subjective interpretations
   - Trivial questions ("What is the first word?")
</instructions>

<passage>
{passage}
</passage>

<format>
Return JSON:
{
  "question": "...",
  "answer": "...",
  "supporting_passage": "..."  // EXACT quote from passage
}
</format>
"""
```

#### 4c. Critical Validation Step

**THIS IS THE MOST IMPORTANT PART**:

```python
def generate_question(passage: str) -> dict | None:
    # ... generate with LLM ...

    # CRITICAL: Validate supporting passage exists verbatim
    if qgen["supporting_passage"] in passage:
        return qgen
    else:
        # Retry if passage doesn't match - LLM hallucinated
        continue
```

This validation ensures:
- The `supporting_passage` is an exact substring of the source
- No hallucinated or paraphrased passages
- Retrieval evaluation will work correctly

#### 4d. Semantic Deduplication

Use LLM to remove duplicate/similar questions:

```python
DEDUP_TEMPLATE = """
Given these questions, identify duplicates (same info, different words).
Return indices of questions to KEEP (unique ones).

Questions:
{numbered_list}

Return: {"indices": [1, 3, 5, ...]}
"""

# Process in batches of 50 for large question sets
# Run final dedup pass across batches
```

#### 4e. Output Format

```python
question_entry = {
    "question": "What was Apple's revenue in Q4 2024?",
    "answer": "Apple reported revenue of $94.9 billion in Q4 2024.",
    "supporting_passage": "For the fourth quarter of fiscal 2024, Apple reported revenue of $94.9 billion, up 6% year over year.",
    "ticker": "AAPL",           # Links to corpus entry
    "company": "Apple Inc.",
    "chunk_index": 12           # Which chunk this came from
}
```

---

### Step 5: Upload to HuggingFace

Two separate configs in the same dataset:

```python
# Corpus
dataset.push_to_hub("chonkie-ai/{benchmark}", config_name="corpus")

# Questions
dataset.push_to_hub("chonkie-ai/{benchmark}", config_name="questions")
```

**Final structure on HuggingFace**:
```
chonkie-ai/ficha
├── corpus/
│   └── train (88 examples, 48.8MB)
└── questions/
    └── train (1,331 QA pairs, 465KB)
```

---

### Quality Checklist

Before publishing any MTCB dataset:

- [ ] All `supporting_passage` values exist verbatim in source documents
- [ ] No duplicate questions (semantic dedup applied)
- [ ] Passages are minimal sufficient context (not too long)
- [ ] Questions are answerable from passage alone
- [ ] Metadata correctly links questions to corpus entries
- [ ] Dataset loads correctly: `load_dataset("chonkie-ai/X", "corpus")`
- [ ] Dataset loads correctly: `load_dataset("chonkie-ai/X", "questions")`

---

## Implementation Roadmap

### Phase 1: Code Benchmark (Gicha/Cocha)
- [ ] Collect top 500 Python repos
- [ ] Extract source files (exclude tests, configs)
- [ ] Generate function/class-level questions
- [ ] Create evaluator mirroring GachaEvaluator
- [ ] Benchmark existing chunkers

### Phase 2: Academic Benchmark (Mocha)
- [ ] Download ArXiv CS papers (open access)
- [ ] Parse PDF → text (or use LaTeX source)
- [ ] Generate section-aware questions
- [ ] Handle citations and references

### Phase 3: Tables Benchmark (Sencha)
- [ ] Extract Wikipedia tables
- [ ] Generate table-specific questions
- [ ] Design table-aware evaluation (row/column matching)

### Phase 4: Remaining Benchmarks
- Legal (Hojicha)
- Dialogue (Chacha)
- Financial (Ficha)

---

## Success Metrics for MTCB

1. **Adoption**: Chunker libraries report MTCB scores
2. **Coverage**: 6+ diverse benchmarks across domains
3. **Reproducibility**: All datasets on HuggingFace, all code open-source
4. **Impact**: Cited in chunking/RAG papers

---

## Open Questions

1. Should we include a **multilingual** benchmark?
2. Should we test **chunk size sensitivity** as a separate dimension?
3. How do we handle **multimodal** documents (images, diagrams)?
4. Should there be a **speed/quality tradeoff** leaderboard?
