#!/usr/bin/env python3
"""Unified Macha benchmark for all Chonkie chunkers.

Tests various chunkers on the Macha dataset (GitHub READMEs) using:
- Voyage 3 Large embeddings via Catsu client
- ChromaDB (in-memory) for vector storage

Metrics calculated:
- HIT@1, HIT@3, HIT@5, HIT@10 (recall at k)
- MRR (Mean Reciprocal Rank)

Usage:
    python macha.py --chunkers fast token
    python macha.py --chunkers all
    python macha.py --chunkers fast --sizes 1024 2048 4096
    python macha.py --list
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import chromadb
import catsu
import numpy as np
from chromadb.config import Settings
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(".env.local")

# Output directory for intermediate files
DATA_DIR = Path(__file__).parent / "data" / "macha_benchmark"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Tokenizer Mapping
# =============================================================================

TOKENIZER_MAP = {
    # Voyage AI models (HuggingFace tokenizers)
    "voyage-3-large": "voyageai/voyage-3-large",
    "voyage-3": "voyageai/voyage-3",
    "voyage-3-lite": "voyageai/voyage-3-lite",
    "voyage-code-3": "voyageai/voyage-code-3",
    "voyage-finance-2": "voyageai/voyage-finance-2",
    "voyage-law-2": "voyageai/voyage-law-2",
    "voyage-code-2": "voyageai/voyage-code-2",
    # OpenAI models (tiktoken)
    "text-embedding-3-large": "cl100k_base",
    "text-embedding-3-small": "cl100k_base",
    "text-embedding-ada-002": "cl100k_base",
    # Cohere models
    "embed-english-v3.0": "Cohere/Cohere-embed-english-v3.0",
    "embed-multilingual-v3.0": "Cohere/Cohere-embed-multilingual-v3.0",
    "embed-english-light-v3.0": "Cohere/Cohere-embed-english-light-v3.0",
    "embed-multilingual-light-v3.0": "Cohere/Cohere-embed-multilingual-light-v3.0",
    # Jina models
    "jina-embeddings-v3": "jinaai/jina-embeddings-v3",
    "jina-embeddings-v2-base-en": "jinaai/jina-embeddings-v2-base-en",
    # Default fallback
    "default": "gpt2",
}


def get_tokenizer_for_model(model: str, override: Optional[str] = None):
    """Get the appropriate tokenizer for an embedding model."""
    if override:
        tokenizer_id = override
    else:
        model_name = model.split(":")[-1] if ":" in model else model
        tokenizer_id = TOKENIZER_MAP.get(model_name, TOKENIZER_MAP["default"])

    if tokenizer_id.startswith("cl100k") or tokenizer_id.startswith("p50k") or tokenizer_id.startswith("r50k"):
        import tiktoken
        return tiktoken.get_encoding(tokenizer_id)
    else:
        from tokenizers import Tokenizer
        return Tokenizer.from_pretrained(tokenizer_id)


# =============================================================================
# Chunker Configurations
# =============================================================================


@dataclass
class ChunkerConfig:
    """Configuration for a chunker."""

    name: str
    description: str
    sizes: List[int]
    unit: str  # "B" for bytes, "T" for tokens
    needs_tokenizer: bool
    needs_embedding_model: bool = False
    needs_special_setup: bool = False
    init_fn: Optional[Callable] = None
    notes: str = ""


def get_chunker_configs() -> Dict[str, ChunkerConfig]:
    """Get all chunker configurations."""
    return {
        "fast": ChunkerConfig(
            name="FastChunker",
            description="SIMD-accelerated byte-based chunker using memchunk",
            sizes=[1024, 2048, 4096, 8192, 16384],
            unit="B",
            needs_tokenizer=False,
            init_fn=lambda size, **_: _init_fast_chunker(size),
        ),
        "token": ChunkerConfig(
            name="TokenChunker",
            description="Simple token-based chunker",
            sizes=[256, 512, 1024, 2048, 4096],
            unit="T",
            needs_tokenizer=True,
            init_fn=lambda size, tokenizer, **_: _init_token_chunker(size, tokenizer),
        ),
        "sentence": ChunkerConfig(
            name="SentenceChunker",
            description="Sentence-aware chunker that respects sentence boundaries",
            sizes=[256, 512, 1024, 2048, 4096],
            unit="T",
            needs_tokenizer=True,
            init_fn=lambda size, tokenizer, **_: _init_sentence_chunker(size, tokenizer),
        ),
        "recursive": ChunkerConfig(
            name="RecursiveChunker",
            description="Recursive chunker with hierarchical splitting rules",
            sizes=[256, 512, 1024, 2048, 4096],
            unit="T",
            needs_tokenizer=True,
            init_fn=lambda size, tokenizer, **_: _init_recursive_chunker(size, tokenizer),
        ),
        "semantic": ChunkerConfig(
            name="SemanticChunker",
            description="Semantic similarity-based chunker",
            sizes=[512, 1024, 2048],
            unit="T",
            needs_tokenizer=False,
            needs_embedding_model=True,
            init_fn=lambda size, **_: _init_semantic_chunker(size),
            notes="Uses internal embedding model for similarity",
        ),
        "late": ChunkerConfig(
            name="LateChunker",
            description="Late interaction chunker using ColBERT-style embeddings",
            sizes=[512, 1024, 2048],
            unit="T",
            needs_tokenizer=False,
            needs_embedding_model=True,
            init_fn=lambda size, **_: _init_late_chunker(size),
            notes="Uses late interaction embeddings",
        ),
        "code": ChunkerConfig(
            name="CodeChunker",
            description="AST-aware chunker for source code",
            sizes=[512, 1024, 2048, 4096],
            unit="T",
            needs_tokenizer=True,
            needs_special_setup=True,
            init_fn=lambda size, tokenizer, **_: _init_code_chunker(size, tokenizer),
            notes="Best for code; requires tree-sitter",
        ),
        "markdown": ChunkerConfig(
            name="MarkdownChunker",
            description="Markdown-aware chunker that respects document structure",
            sizes=[256, 512, 1024, 2048, 4096],
            unit="T",
            needs_tokenizer=True,
            init_fn=lambda size, tokenizer, **_: _init_markdown_chunker(size, tokenizer),
            notes="Optimized for markdown documents",
        ),
        "neural": ChunkerConfig(
            name="NeuralChunker",
            description="Neural model-based chunker using learned boundaries",
            sizes=[],
            unit="",
            needs_tokenizer=False,
            needs_special_setup=True,
            init_fn=lambda **_: _init_neural_chunker(),
            notes="Uses trained model; no size parameter",
        ),
        "slumber": ChunkerConfig(
            name="SlumberChunker",
            description="LLM-powered chunker using a 'genie'",
            sizes=[512, 1024, 2048],
            unit="T",
            needs_tokenizer=True,
            needs_special_setup=True,
            init_fn=None,
            notes="Requires LLM/genie configuration",
        ),
    }


# =============================================================================
# Chunker Initialization Functions
# =============================================================================


def _init_fast_chunker(size: int):
    from chonkie import FastChunker
    return FastChunker(
        chunk_size=size,
        delimiters="\n.!?",
        forward_fallback=True,
    )


def _init_token_chunker(size: int, tokenizer):
    from chonkie import TokenChunker
    return TokenChunker(tokenizer=tokenizer, chunk_size=size, chunk_overlap=0)


def _init_sentence_chunker(size: int, tokenizer):
    from chonkie import SentenceChunker
    return SentenceChunker(tokenizer=tokenizer, chunk_size=size, chunk_overlap=0)


def _init_recursive_chunker(size: int, tokenizer):
    from chonkie import RecursiveChunker
    return RecursiveChunker(tokenizer=tokenizer, chunk_size=size)


def _init_semantic_chunker(size: int):
    from chonkie import SemanticChunker
    return SemanticChunker(chunk_size=size)


def _init_late_chunker(size: int):
    from chonkie import LateChunker
    return LateChunker(chunk_size=size)


def _init_code_chunker(size: int, tokenizer):
    from chonkie import CodeChunker
    return CodeChunker(tokenizer=tokenizer, chunk_size=size)


def _init_markdown_chunker(size: int, tokenizer):
    from chonkie import MarkdownChunker
    return MarkdownChunker(tokenizer=tokenizer, chunk_size=size)


def _init_neural_chunker():
    from chonkie import NeuralChunker
    return NeuralChunker()


# =============================================================================
# Utility Functions
# =============================================================================


def log(msg: str):
    """Print with flush for immediate output."""
    print(msg, flush=True)


def save_jsonl(path: Path, data: List[Dict]):
    """Save list of dicts to JSONL file."""
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    log(f"    Saved: {path}")


def load_jsonl(path: Path) -> List[Dict]:
    """Load JSONL file to list of dicts."""
    with open(path) as f:
        return [json.loads(line) for line in f]


def save_embeddings(path: Path, embeddings: List[List[float]]):
    """Save embeddings as numpy array."""
    np.save(path, np.array(embeddings))
    log(f"    Saved: {path}")


def load_embeddings(path: Path) -> List[List[float]]:
    """Load embeddings from numpy file."""
    return np.load(path).tolist()


def calculate_mrr(ranks: List[int]) -> float:
    """Calculate Mean Reciprocal Rank."""
    reciprocal_ranks = [1.0 / rank if rank > 0 else 0.0 for rank in ranks]
    return float(np.mean(reciprocal_ranks))


def calculate_batch_size(chunk_size_tokens: int, max_tokens_per_batch: int = 100_000) -> int:
    """Calculate optimal batch size based on chunk size."""
    return max(1, min(128, max_tokens_per_batch // max(chunk_size_tokens, 1)))


def get_embeddings_batched(
    client: catsu.Client,
    texts: List[str],
    model: str = "voyage-3-large",
    input_type: str = "document",
    batch_size: int = 128,
) -> List[List[float]]:
    """Get embeddings in batches for efficiency."""
    all_embeddings = []

    for i in tqdm(
        range(0, len(texts), batch_size), desc=f"Embedding ({input_type})", leave=False
    ):
        batch = texts[i : i + batch_size]
        response = client.embed(
            model=model,
            input=batch,
            input_type=input_type,
        )
        all_embeddings.extend(response.embeddings)

    return all_embeddings


def get_model_short_name(model: str) -> str:
    """Get a short name for the model for use in filenames."""
    if ":" in model:
        model = model.split(":")[-1]
    return model.replace("/", "-").replace("_", "-")


# =============================================================================
# Data Loading
# =============================================================================


def load_or_download_corpus() -> tuple[List[Dict], List[Dict]]:
    """Load corpus from cache or download from HuggingFace."""
    corpus_path = DATA_DIR / "macha_corpus.jsonl"
    questions_path = DATA_DIR / "macha_questions.jsonl"

    if corpus_path.exists() and questions_path.exists():
        log("  Loading cached corpus and questions...")
        corpus = load_jsonl(corpus_path)
        questions = load_jsonl(questions_path)
        log(f"    Loaded {len(corpus)} READMEs and {len(questions)} questions from cache")
    else:
        log("  Downloading from HuggingFace...")
        from datasets import load_dataset

        corpus_ds = load_dataset("chonkie-ai/macha", "corpus", split="train")
        questions_ds = load_dataset("chonkie-ai/macha", "questions", split="train")

        corpus = [
            {
                "repo": c["repo"],
                "text": c["readme"],
                "stars": c["stars"],
                "rank": c["rank"],
                "token_count": c["token-count"],
            }
            for c in corpus_ds
        ]
        questions = [
            {
                "repo": q["repo"],
                "question": q["question"],
                "answer": q["answer"],
                "chunk-must-contain": q["chunk-must-contain"],
            }
            for q in questions_ds
        ]

        save_jsonl(corpus_path, corpus)
        save_jsonl(questions_path, questions)
        log(f"    Downloaded {len(corpus)} READMEs and {len(questions)} questions")

    return corpus, questions


def load_or_embed_questions(
    catsu_client: catsu.Client,
    questions: List[Dict],
    model: str = "voyage-3-large",
) -> List[List[float]]:
    """Embed all questions, with caching."""
    model_short = get_model_short_name(model)
    embeddings_path = DATA_DIR / f"question_embeddings_{model_short}.npy"

    if embeddings_path.exists():
        log(f"  Loading cached question embeddings ({model_short})...")
        embeddings = load_embeddings(embeddings_path)
        log(f"    Loaded {len(embeddings)} embeddings from cache")
        return embeddings

    log(f"  Embedding {len(questions)} questions with {model}...")
    question_texts = [q["question"] for q in questions]
    embeddings = get_embeddings_batched(
        catsu_client, question_texts, model=model, input_type="query"
    )
    save_embeddings(embeddings_path, embeddings)
    return embeddings


# =============================================================================
# Chunking and Embedding
# =============================================================================


def chunk_corpus(
    corpus: List[Dict],
    chunker: Any,
    chunker_key: str,
    size: int,
    unit: str,
    suffix: str = "",
) -> List[Dict]:
    """Chunk all READMEs and return list of chunk dicts."""
    chunks_path = DATA_DIR / f"chunks_{chunker_key}_{size}{unit}{suffix}.jsonl"

    if chunks_path.exists():
        log(f"  Loading cached chunks for {chunker_key} {size}{unit}...")
        chunks = load_jsonl(chunks_path)
        chunks = [c for c in chunks if c["text"].strip()]
        log(f"    Loaded {len(chunks)} chunks from cache")
        return chunks

    log(f"  Chunking corpus with {chunker_key} at {size}{unit}...")

    chunks = []
    skipped = 0
    for doc in tqdm(corpus, desc="Chunking", leave=False):
        # Skip documents with None or empty text
        if not doc.get("text"):
            skipped += 1
            continue
        doc_chunks = chunker.chunk(doc["text"])
        for i, chunk in enumerate(doc_chunks):
            if not chunk.text.strip():
                skipped += 1
                continue
            chunks.append(
                {
                    "repo": doc["repo"],
                    "chunk_index": i,
                    "text": chunk.text,
                    "start_index": chunk.start_index,
                    "end_index": chunk.end_index,
                    "token_count": getattr(chunk, "token_count", 0),
                }
            )

    save_jsonl(chunks_path, chunks)
    log(f"    Created {len(chunks)} chunks (skipped {skipped} empty)")
    return chunks


def embed_chunks(
    catsu_client: catsu.Client,
    chunks: List[Dict],
    chunker_key: str,
    size: int,
    unit: str,
    model: str = "voyage-3-large",
    suffix: str = "",
) -> List[List[float]]:
    """Embed all chunks, with caching."""
    model_short = get_model_short_name(model)
    embeddings_path = DATA_DIR / f"chunk_embeddings_{chunker_key}_{size}{unit}{suffix}_{model_short}.npy"

    if embeddings_path.exists():
        log(f"  Loading cached chunk embeddings for {chunker_key} {size}{unit} ({model_short})...")
        embeddings = load_embeddings(embeddings_path)
        log(f"    Loaded {len(embeddings)} embeddings from cache")
        return embeddings

    if unit == "T":
        batch_size = calculate_batch_size(size)
    elif unit == "B":
        # More conservative: ~3 bytes per token for markdown
        batch_size = calculate_batch_size(size // 3)
    else:
        batch_size = 64

    # Cap batch size to avoid API limits (120k token limit)
    batch_size = min(batch_size, 32)

    log(f"  Embedding {len(chunks)} chunks with {model} (batch_size={batch_size})...")
    chunk_texts = [c["text"] for c in chunks]
    embeddings = get_embeddings_batched(
        catsu_client, chunk_texts, model=model, input_type="document", batch_size=batch_size
    )
    save_embeddings(embeddings_path, embeddings)
    return embeddings


# =============================================================================
# Benchmarking
# =============================================================================


def benchmark_configuration(
    chunker_key: str,
    size: int,
    unit: str,
    chunks: List[Dict],
    chunk_embeddings: List[List[float]],
    questions: List[Dict],
    question_embeddings: List[List[float]],
    k_values: List[int],
    model: str = "voyage-3-large",
) -> Dict[str, Any]:
    """Benchmark a single chunker configuration."""
    log(f"\n{'='*60}")
    log(f"Evaluating: {chunker_key} @ {size}{unit}")
    log(f"{'='*60}")

    start_time = time.time()

    # Create ChromaDB collection
    log("\n  Creating ChromaDB collection...")
    chroma_client = chromadb.Client(
        Settings(is_persistent=False, anonymized_telemetry=False)
    )
    collection = chroma_client.create_collection(
        name=f"macha_{chunker_key}_{size}{unit}",
        metadata={"hnsw:space": "cosine"},
    )

    # Add chunks in batches
    chroma_batch_size = 5000
    for i in tqdm(
        range(0, len(chunks), chroma_batch_size), desc="Storing", leave=False
    ):
        end_idx = min(i + chroma_batch_size, len(chunks))
        collection.add(
            embeddings=chunk_embeddings[i:end_idx],
            documents=[c["text"] for c in chunks[i:end_idx]],
            metadatas=[
                {"repo": c["repo"], "token_count": c.get("token_count", 0)}
                for c in chunks[i:end_idx]
            ],
            ids=[f"chunk_{j}" for j in range(i, end_idx)],
        )

    store_time = time.time() - start_time
    log(f"    Storage time: {store_time:.2f}s")

    # Evaluate retrieval
    log("\n  Evaluating retrieval...")
    eval_start = time.time()

    max_k = max(k_values)
    hits = {k: 0 for k in k_values}
    ranks = []
    results_detail = []

    for i, (q, q_emb) in enumerate(
        tqdm(
            zip(questions, question_embeddings),
            total=len(questions),
            desc="Querying",
            leave=False,
        )
    ):
        relevant_passage = q["chunk-must-contain"]

        results = collection.query(
            query_embeddings=[q_emb],
            n_results=max_k,
        )

        retrieved_docs = results["documents"][0] if results["documents"] else []

        # Find rank of first relevant chunk
        first_relevant_rank = 0
        for rank, doc in enumerate(retrieved_docs, start=1):
            if relevant_passage in doc:
                first_relevant_rank = rank
                break

        ranks.append(first_relevant_rank)

        for k in k_values:
            if any(relevant_passage in doc for doc in retrieved_docs[:k]):
                hits[k] += 1

        # Save detail for analysis
        results_detail.append(
            {
                "question_idx": i,
                "question": q["question"],
                "repo": q["repo"],
                "expected_passage": relevant_passage[:100],
                "rank": first_relevant_rank,
                "found": first_relevant_rank > 0,
            }
        )

    eval_time = time.time() - eval_start
    total_time = time.time() - start_time

    # Save detailed results
    results_path = DATA_DIR / f"results_{chunker_key}_{size}{unit}.jsonl"
    save_jsonl(results_path, results_detail)

    # Calculate metrics
    total_questions = len(questions)
    mrr = calculate_mrr(ranks)
    hit_rates = {k: hits[k] / total_questions for k in k_values}

    # Calculate average token count if available
    token_counts = [c.get("token_count", 0) for c in chunks]
    avg_tokens = sum(token_counts) / len(chunks) if token_counts else 0

    log(f"\n  Results:")
    log(f"    Total chunks: {len(chunks)}")
    if avg_tokens > 0:
        log(f"    Avg tokens/chunk: {avg_tokens:.1f}")
    log(f"    Total questions: {total_questions}")
    log(f"    Eval time: {eval_time:.2f}s")
    log("")
    for k in k_values:
        log(f"    HIT@{k:2d}: {hit_rates[k]:.4f} ({hits[k]}/{total_questions})")
    log(f"    MRR:    {mrr:.4f}")

    return {
        "chunker": chunker_key,
        "size": size,
        "unit": unit,
        "model": model,
        "total_chunks": len(chunks),
        "avg_tokens": avg_tokens,
        "total_questions": total_questions,
        "total_time": total_time,
        "hit_rates": hit_rates,
        "mrr": mrr,
        "hits": hits,
    }


# =============================================================================
# Main
# =============================================================================


def print_chunker_list():
    """Print available chunkers and their configurations."""
    configs = get_chunker_configs()

    print("\nAvailable Chunkers:")
    print("=" * 80)

    for key, config in configs.items():
        status = ""
        if config.needs_special_setup:
            status = " [SPECIAL SETUP]"
        elif config.needs_embedding_model:
            status = " [NEEDS EMBEDDING MODEL]"

        print(f"\n  {key}: {config.name}{status}")
        print(f"      {config.description}")
        if config.sizes:
            print(f"      Default sizes: {config.sizes} {config.unit}")
        if config.notes:
            print(f"      Note: {config.notes}")

    print("\n" + "=" * 80)
    print("\nStandard chunkers (ready to benchmark):")
    print("  fast, token, sentence, recursive, markdown")
    print("\nAdvanced chunkers (require additional setup):")
    print("  semantic, late, code, neural, slumber")


def main():
    """Main benchmark function."""
    parser = argparse.ArgumentParser(
        description="Macha benchmark for Chonkie chunkers (Markdown/README corpus)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python macha.py --chunkers fast token          # Benchmark FastChunker and TokenChunker
  python macha.py --chunkers all                 # Benchmark all standard chunkers
  python macha.py --chunkers markdown --sizes 512 1024  # Test MarkdownChunker
  python macha.py --chunkers fast --model text-embedding-3-large  # Use OpenAI embeddings
  python macha.py --list                         # List all available chunkers
        """,
    )
    parser.add_argument(
        "--chunkers",
        nargs="+",
        default=["fast"],
        help="Chunkers to benchmark (fast, token, sentence, recursive, markdown, semantic, late, code, or 'all')",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        help="Override default chunk sizes (applies to all selected chunkers)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available chunkers and exit",
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=[1, 3, 5, 10],
        help="K values for HIT@k metrics (default: 1 3 5 10)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="voyage-3-large",
        help="Embedding model to use via Catsu (default: voyage-3-large). "
             "Examples: voyage-3-large, text-embedding-3-large, openai:text-embedding-3-small",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Override tokenizer for chunking (auto-detected from model by default). "
             "Examples: voyageai/voyage-3-large, cl100k_base, gpt2",
    )
    parser.add_argument(
        "--questions-only",
        action="store_true",
        help="Only process repos that have questions (127 repos instead of 957)",
    )

    args = parser.parse_args()

    if args.list:
        print_chunker_list()
        return

    configs = get_chunker_configs()

    # Determine which chunkers to run
    if "all" in args.chunkers:
        # Standard chunkers that don't need special setup
        chunker_keys = ["fast", "token", "sentence", "recursive", "markdown"]
    else:
        chunker_keys = args.chunkers

    # Validate chunker selections
    for key in chunker_keys:
        if key not in configs:
            print(f"Error: Unknown chunker '{key}'")
            print(f"Available: {', '.join(configs.keys())}")
            return

    log("=" * 60)
    log("Macha Benchmark - Chonkie Chunkers (Markdown/README)")
    log("=" * 60)
    log(f"\nData directory: {DATA_DIR}")

    # Determine tokenizer ID for display
    tokenizer_display = args.tokenizer or TOKENIZER_MAP.get(
        args.model.split(":")[-1], TOKENIZER_MAP["default"]
    )

    log("\nConfiguration:")
    log(f"  Chunkers: {chunker_keys}")
    log(f"  Embedding: {args.model} (via Catsu)")
    log(f"  Tokenizer: {tokenizer_display}")
    log(f"  Vector DB: ChromaDB (in-memory)")
    log(f"  K values: {args.k_values}")

    # Initialize tokenizer if needed
    tokenizer = None
    needs_tokenizer = any(configs[k].needs_tokenizer for k in chunker_keys)
    if needs_tokenizer:
        tokenizer_id = args.tokenizer or TOKENIZER_MAP.get(
            args.model.split(":")[-1], TOKENIZER_MAP["default"]
        )
        log(f"\nLoading tokenizer: {tokenizer_id}...")
        tokenizer = get_tokenizer_for_model(args.model, override=args.tokenizer)
        log("  Done")

    # Initialize Catsu client
    log("\nInitializing Catsu client...")
    catsu_client = catsu.Client()
    log("  Done")

    # Load corpus
    log("\n" + "-" * 60)
    log("STEP 1: Loading corpus")
    log("-" * 60)
    corpus, questions = load_or_download_corpus()

    # Filter to only repos with questions if requested
    if args.questions_only:
        repos_with_questions = set(q["repo"] for q in questions)
        corpus = [c for c in corpus if c["repo"] in repos_with_questions]
        log(f"  Filtered to {len(corpus)} repos with questions")

    # Embed questions
    log("\n" + "-" * 60)
    log("STEP 2: Embedding questions")
    log("-" * 60)
    question_embeddings = load_or_embed_questions(catsu_client, questions, model=args.model)

    # Run benchmarks
    all_results = []

    for chunker_key in chunker_keys:
        config = configs[chunker_key]

        if config.init_fn is None:
            log(f"\nSkipping {chunker_key}: requires special setup")
            continue

        sizes = args.sizes if args.sizes else config.sizes
        if not sizes:
            log(f"\nSkipping {chunker_key}: no sizes configured")
            continue

        log("\n" + "-" * 60)
        log(f"STEP 3: Benchmarking {config.name}")
        log("-" * 60)
        log(f"  Sizes: {sizes} {config.unit}")

        # Suffix for questions-only mode
        cache_suffix = "_qonly" if args.questions_only else ""

        for size in sizes:
            # Initialize chunker
            try:
                chunker = config.init_fn(size=size, tokenizer=tokenizer)
            except Exception as e:
                log(f"\n  Error initializing {chunker_key} at {size}{config.unit}: {e}")
                continue

            # Chunk corpus
            chunks = chunk_corpus(corpus, chunker, chunker_key, size, config.unit, suffix=cache_suffix)

            # Embed chunks
            chunk_embeddings = embed_chunks(
                catsu_client, chunks, chunker_key, size, config.unit, model=args.model, suffix=cache_suffix
            )

            # Benchmark
            result = benchmark_configuration(
                chunker_key=chunker_key,
                size=size,
                unit=config.unit,
                chunks=chunks,
                chunk_embeddings=chunk_embeddings,
                questions=questions,
                question_embeddings=question_embeddings,
                k_values=args.k_values,
                model=args.model,
            )
            all_results.append(result)

    if not all_results:
        log("\nNo results to report.")
        return

    # Save summary
    summary_path = DATA_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log(f"\nSaved summary: {summary_path}")

    # Print comparative summary
    log("\n" + "=" * 100)
    log("COMPARATIVE SUMMARY")
    log("=" * 100)
    log("")

    header = f"{'Chunker':>12} | {'Size':>8} | {'Chunks':>8} | {'HIT@1':>8} | {'HIT@3':>8} | {'HIT@5':>8} | {'HIT@10':>8} | {'MRR':>8}"
    log(header)
    log("-" * len(header))

    for r in all_results:
        size_str = f"{r['size']}{r['unit']}"
        log(
            f"{r['chunker']:>12} | {size_str:>8} | {r['total_chunks']:>8} | "
            f"{r['hit_rates'][1]:>8.4f} | {r['hit_rates'][3]:>8.4f} | "
            f"{r['hit_rates'][5]:>8.4f} | {r['hit_rates'][10]:>8.4f} | {r['mrr']:>8.4f}"
        )

    log("")
    log("Best configurations:")
    log("-" * 50)

    for metric in ["HIT@1", "HIT@3", "HIT@5", "HIT@10", "MRR"]:
        if metric == "MRR":
            best = max(all_results, key=lambda x: x["mrr"])
            val = best["mrr"]
        else:
            k = int(metric.split("@")[1])
            best = max(all_results, key=lambda x: x["hit_rates"][k])
            val = best["hit_rates"][k]
        log(f"  {metric:8s}: {best['chunker']} @ {best['size']}{best['unit']} ({val:.4f})")

    log("\n" + "=" * 60)
    log("BENCHMARK COMPLETE")
    log("=" * 60)
    log(f"\nAll outputs saved to: {DATA_DIR}")


if __name__ == "__main__":
    main()
