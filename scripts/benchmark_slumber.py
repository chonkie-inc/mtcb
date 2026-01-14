#!/usr/bin/env python3
"""SlumberChunker benchmark on Macha dataset using Cerebras.

Tests SlumberChunker (LLM-powered) vs RecursiveChunker on GitHub READMEs.

Usage:
    python benchmark_slumber.py                  # Run on all repos
    python benchmark_slumber.py --limit 10      # Run on first 10 repos
    python benchmark_slumber.py --chunk-size 1024
"""

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import catsu
import chromadb
import numpy as np
from chromadb.config import Settings
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(".env.local")

# Output directory
DATA_DIR = Path(__file__).parent / "data" / "slumber_benchmark"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Tokenizer for Voyage models
TOKENIZER_ID = "voyageai/voyage-3-large"


def log(msg: str):
    """Print with flush."""
    print(msg, flush=True)


def load_corpus() -> tuple[List[Dict], List[Dict]]:
    """Load Macha corpus from cache or HuggingFace."""
    corpus_path = DATA_DIR.parent / "macha_benchmark" / "macha_corpus.jsonl"
    questions_path = DATA_DIR.parent / "macha_benchmark" / "macha_questions.jsonl"

    if corpus_path.exists() and questions_path.exists():
        log("Loading cached corpus...")
        with open(corpus_path) as f:
            corpus = [json.loads(line) for line in f]
        with open(questions_path) as f:
            questions = [json.loads(line) for line in f]
    else:
        log("Downloading from HuggingFace...")
        from datasets import load_dataset

        corpus_ds = load_dataset("chonkie-ai/macha", "corpus", split="train")
        questions_ds = load_dataset("chonkie-ai/macha", "questions", split="train")

        corpus = [
            {
                "repo": c["repo"],
                "text": c["readme"],
                "stars": c["stars"],
            }
            for c in corpus_ds
        ]
        questions = [
            {
                "repo": q["repo"],
                "question": q["question"],
                "chunk-must-contain": q["chunk-must-contain"],
            }
            for q in questions_ds
        ]

    return corpus, questions


def get_embeddings_batched(
    client: catsu.Client,
    texts: List[str],
    model: str = "voyage-3-large",
    input_type: str = "document",
    batch_size: int = 32,
) -> List[List[float]]:
    """Embed texts in batches."""
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Embedding ({input_type})", leave=False):
        batch = texts[i : i + batch_size]
        response = client.embed(model=model, input=batch, input_type=input_type)
        all_embeddings.extend(response.embeddings)
    return all_embeddings


def calculate_mrr(ranks: List[int]) -> float:
    """Calculate Mean Reciprocal Rank."""
    reciprocal_ranks = [1.0 / rank if rank > 0 else 0.0 for rank in ranks]
    return float(np.mean(reciprocal_ranks))


@dataclass
class BenchmarkResult:
    """Result from benchmarking a chunker."""
    chunker_name: str
    chunk_size: int
    num_chunks: int
    avg_tokens: float
    hit_rates: Dict[int, float]
    mrr: float
    chunk_time: float
    eval_time: float
    hits: Dict[int, int]
    total_questions: int


class ParallelChunker:
    """Handles parallel chunking of documents with SlumberChunker."""

    def __init__(self, tokenizer, chunk_size: int, cerebras_key: str, llm_model: str):
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.cerebras_key = cerebras_key
        self.llm_model = llm_model
        self.chunks_lock = threading.Lock()
        self.print_lock = threading.Lock()
        self.all_chunks = []
        self.processed = 0
        self.skipped = 0

    def _create_chunker(self):
        """Create a new SlumberChunker instance for this thread."""
        from chonkie import SlumberChunker
        from chonkie.genie import OpenAIGenie

        genie = OpenAIGenie(
            model=self.llm_model,
            base_url="https://api.cerebras.ai/v1",
            api_key=self.cerebras_key,
        )
        return SlumberChunker(
            genie=genie,
            tokenizer=self.tokenizer,
            chunk_size=self.chunk_size,
        )

    def process_doc(self, doc: Dict) -> List[Dict]:
        """Process a single document. Thread-safe."""
        if not doc.get("text"):
            with self.chunks_lock:
                self.skipped += 1
            return []

        try:
            # Each thread gets its own chunker instance
            chunker = self._create_chunker()
            doc_chunks = chunker.chunk(doc["text"])

            chunks = []
            for i, chunk in enumerate(doc_chunks):
                if not chunk.text.strip():
                    continue
                chunks.append({
                    "repo": doc["repo"],
                    "chunk_index": i,
                    "text": chunk.text,
                    "token_count": getattr(chunk, "token_count", 0),
                })

            with self.chunks_lock:
                self.all_chunks.extend(chunks)
                self.processed += 1

            return chunks

        except Exception as e:
            with self.print_lock:
                log(f"    Error chunking {doc['repo']}: {e}")
            with self.chunks_lock:
                self.skipped += 1
            return []


def chunk_corpus_with_chunker(
    chunker,
    corpus: List[Dict],
    chunker_name: str,
) -> tuple[List[Dict], float]:
    """Chunk entire corpus sequentially and return chunks with timing."""
    log(f"\n  Chunking corpus with {chunker_name}...")

    all_chunks = []
    chunk_start = time.time()
    skipped = 0

    for doc in tqdm(corpus, desc="Chunking", leave=False):
        if not doc.get("text"):
            skipped += 1
            continue
        try:
            doc_chunks = chunker.chunk(doc["text"])
            for i, chunk in enumerate(doc_chunks):
                if not chunk.text.strip():
                    continue
                all_chunks.append({
                    "repo": doc["repo"],
                    "chunk_index": i,
                    "text": chunk.text,
                    "token_count": getattr(chunk, "token_count", 0),
                })
        except Exception as e:
            log(f"    Error chunking {doc['repo']}: {e}")
            skipped += 1
            continue

    chunk_time = time.time() - chunk_start
    log(f"    Created {len(all_chunks)} chunks in {chunk_time:.2f}s (skipped {skipped})")

    return all_chunks, chunk_time


def chunk_corpus_parallel(
    corpus: List[Dict],
    tokenizer,
    chunk_size: int,
    cerebras_key: str,
    llm_model: str,
    workers: int = 8,
) -> tuple[List[Dict], float]:
    """Chunk corpus in parallel using multiple workers."""
    log(f"\n  Chunking corpus with SlumberChunker ({workers} workers)...")

    parallel_chunker = ParallelChunker(tokenizer, chunk_size, cerebras_key, llm_model)

    chunk_start = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(parallel_chunker.process_doc, doc): doc["repo"] for doc in corpus}

        # Progress bar
        with tqdm(total=len(corpus), desc="Chunking", leave=False) as pbar:
            for future in as_completed(futures):
                pbar.update(1)

    chunk_time = time.time() - chunk_start
    log(f"    Created {len(parallel_chunker.all_chunks)} chunks in {chunk_time:.2f}s")
    log(f"    Processed: {parallel_chunker.processed}, Skipped: {parallel_chunker.skipped}")

    return parallel_chunker.all_chunks, chunk_time


def benchmark_chunker(
    chunks: List[Dict],
    chunker_name: str,
    chunk_size: int,
    chunk_time: float,
    questions: List[Dict],
    catsu_client: catsu.Client,
    k_values: List[int] = [1, 3, 5, 10],
    model: str = "voyage-3-large",
) -> BenchmarkResult:
    """Benchmark a chunker given pre-computed chunks."""
    log(f"\n{'='*60}")
    log(f"Evaluating: {chunker_name} @ {chunk_size} tokens")
    log(f"{'='*60}")

    # Calculate avg tokens
    token_counts = [c.get("token_count", 0) for c in chunks]
    avg_tokens = sum(token_counts) / len(chunks) if chunks else 0
    log(f"  Total chunks: {len(chunks)}")
    log(f"  Avg tokens/chunk: {avg_tokens:.1f}")

    # Embed chunks
    log("\n  Embedding chunks...")
    chunk_texts = [c["text"] for c in chunks]
    chunk_embeddings = get_embeddings_batched(
        catsu_client, chunk_texts, model=model, input_type="document"
    )

    # Embed questions
    log("\n  Embedding questions...")
    question_texts = [q["question"] for q in questions]
    question_embeddings = get_embeddings_batched(
        catsu_client, question_texts, model=model, input_type="query"
    )

    # Create ChromaDB collection
    log("\n  Creating vector store...")
    chroma_client = chromadb.Client(Settings(is_persistent=False, anonymized_telemetry=False))
    collection = chroma_client.create_collection(
        name=f"slumber_macha_{chunker_name}",
        metadata={"hnsw:space": "cosine"},
    )

    # Add chunks in batches
    batch_size = 5000
    for i in range(0, len(chunks), batch_size):
        end_idx = min(i + batch_size, len(chunks))
        collection.add(
            embeddings=chunk_embeddings[i:end_idx],
            documents=[c["text"] for c in chunks[i:end_idx]],
            metadatas=[{"repo": c["repo"]} for c in chunks[i:end_idx]],
            ids=[f"chunk_{j}" for j in range(i, end_idx)],
        )

    # Evaluate
    log("\n  Evaluating retrieval...")
    eval_start = time.time()
    max_k = max(k_values)
    hits = {k: 0 for k in k_values}
    ranks = []

    for q, q_emb in tqdm(zip(questions, question_embeddings), total=len(questions), desc="Querying", leave=False):
        relevant_passage = q["chunk-must-contain"]

        results = collection.query(query_embeddings=[q_emb], n_results=max_k)
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

    eval_time = time.time() - eval_start

    # Calculate metrics
    total_questions = len(questions)
    mrr = calculate_mrr(ranks)
    hit_rates = {k: hits[k] / total_questions for k in k_values}

    log(f"\n  Results:")
    for k in k_values:
        log(f"    HIT@{k:2d}: {hit_rates[k]:.4f} ({hits[k]}/{total_questions})")
    log(f"    MRR:    {mrr:.4f}")

    return BenchmarkResult(
        chunker_name=chunker_name,
        chunk_size=chunk_size,
        num_chunks=len(chunks),
        avg_tokens=avg_tokens,
        hit_rates=hit_rates,
        mrr=mrr,
        chunk_time=chunk_time,
        eval_time=eval_time,
        hits=hits,
        total_questions=total_questions,
    )


def main():
    parser = argparse.ArgumentParser(description="SlumberChunker benchmark on Macha (Cerebras)")
    parser.add_argument("--chunk-size", type=int, default=512, help="Chunk size in tokens (default: 512)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of repos to process")
    parser.add_argument("--model", type=str, default="voyage-3-large", help="Embedding model")
    parser.add_argument("--llm-model", type=str, default="llama-3.3-70b", help="Cerebras model")
    parser.add_argument("--k-values", nargs="+", type=int, default=[1, 3, 5, 10], help="K values for HIT@k")
    parser.add_argument("--skip-recursive", action="store_true", help="Skip RecursiveChunker baseline")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers (default: 8)")

    args = parser.parse_args()

    log("=" * 60)
    log("SlumberChunker vs RecursiveChunker on Macha (Cerebras)")
    log("=" * 60)

    # Check for Cerebras API key
    cerebras_key = os.environ.get("CEREBRAS_API_KEY")
    if not cerebras_key:
        log("Error: CEREBRAS_API_KEY not set in environment")
        return

    # Load corpus
    log("\nLoading corpus...")
    corpus, questions = load_corpus()

    # Filter out None texts
    corpus = [c for c in corpus if c.get("text")]

    # Only keep repos that have questions (optimization - 127 instead of 957)
    repos_with_questions = set(q["repo"] for q in questions)
    corpus = [c for c in corpus if c["repo"] in repos_with_questions]

    if args.limit:
        corpus = corpus[:args.limit]
        # Filter questions to only those repos
        repo_set = set(c["repo"] for c in corpus)
        questions = [q for q in questions if q["repo"] in repo_set]

    log(f"  Corpus: {len(corpus)} READMEs (filtered to repos with questions)")
    log(f"  Questions: {len(questions)}")

    log(f"\nConfiguration:")
    log(f"  Chunk size: {args.chunk_size} tokens")
    log(f"  Embedding model: {args.model}")
    log(f"  LLM model: {args.llm_model} (Cerebras)")
    log(f"  Workers: {args.workers}")

    # Initialize tokenizer
    log("\nLoading tokenizer...")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_pretrained(TOKENIZER_ID)

    # Initialize Catsu client
    log("Initializing Catsu client...")
    catsu_client = catsu.Client()

    results = []

    # Benchmark RecursiveChunker first (as baseline)
    if not args.skip_recursive:
        log("\n" + "-" * 60)
        log("Setting up RecursiveChunker (baseline)...")
        from chonkie import RecursiveChunker

        recursive_chunker = RecursiveChunker(tokenizer=tokenizer, chunk_size=args.chunk_size)
        recursive_chunks, recursive_chunk_time = chunk_corpus_with_chunker(
            recursive_chunker, corpus, "RecursiveChunker"
        )

        recursive_result = benchmark_chunker(
            chunks=recursive_chunks,
            chunker_name="RecursiveChunker",
            chunk_size=args.chunk_size,
            chunk_time=recursive_chunk_time,
            questions=questions,
            catsu_client=catsu_client,
            k_values=args.k_values,
            model=args.model,
        )
        results.append(recursive_result)

    # Benchmark SlumberChunker with Cerebras (parallel)
    log("\n" + "-" * 60)
    log(f"Setting up SlumberChunker (Cerebras, {args.workers} workers)...")

    slumber_chunks, slumber_chunk_time = chunk_corpus_parallel(
        corpus=corpus,
        tokenizer=tokenizer,
        chunk_size=args.chunk_size,
        cerebras_key=cerebras_key,
        llm_model=args.llm_model,
        workers=args.workers,
    )

    # Save chunks for analysis
    chunks_path = DATA_DIR / f"chunks_slumber_{args.chunk_size}T.jsonl"
    with open(chunks_path, "w") as f:
        for c in slumber_chunks:
            f.write(json.dumps(c) + "\n")
    log(f"  Saved chunks to: {chunks_path}")

    slumber_result = benchmark_chunker(
        chunks=slumber_chunks,
        chunker_name="SlumberChunker",
        chunk_size=args.chunk_size,
        chunk_time=slumber_chunk_time,
        questions=questions,
        catsu_client=catsu_client,
        k_values=args.k_values,
        model=args.model,
    )
    results.append(slumber_result)

    # Print comparative summary
    log("\n" + "=" * 80)
    log("COMPARATIVE SUMMARY")
    log("=" * 80)
    log(f"\nCorpus: {len(corpus)} READMEs, {len(questions)} questions")
    log("")

    header = f"{'Chunker':>18} | {'Chunks':>8} | {'AvgTok':>8} | {'HIT@1':>8} | {'HIT@3':>8} | {'HIT@5':>8} | {'HIT@10':>8} | {'MRR':>8} | {'Time':>8}"
    log(header)
    log("-" * len(header))

    for r in results:
        log(
            f"{r.chunker_name:>18} | {r.num_chunks:>8} | {r.avg_tokens:>8.1f} | "
            f"{r.hit_rates[1]:>8.4f} | {r.hit_rates[3]:>8.4f} | "
            f"{r.hit_rates[5]:>8.4f} | {r.hit_rates[10]:>8.4f} | {r.mrr:>8.4f} | {r.chunk_time:>7.1f}s"
        )

    # Calculate improvement if we have both
    if len(results) == 2:
        recursive_result = results[0]
        slumber_result = results[1]

        log("\n" + "-" * 60)
        log("COMPARISON (SlumberChunker vs RecursiveChunker)")
        log("-" * 60)

        for metric_name, metric_key in [("HIT@1", 1), ("HIT@3", 3), ("HIT@5", 5), ("HIT@10", 10), ("MRR", "mrr")]:
            if metric_key == "mrr":
                slumber_val = slumber_result.mrr
                recursive_val = recursive_result.mrr
            else:
                slumber_val = slumber_result.hit_rates[metric_key]
                recursive_val = recursive_result.hit_rates[metric_key]

            diff = slumber_val - recursive_val
            diff_pp = diff * 100
            sign = "+" if diff >= 0 else ""
            winner = "SlumberChunker" if diff > 0 else "RecursiveChunker" if diff < 0 else "TIE"

            log(f"  {metric_name:8s}: {slumber_val:.4f} vs {recursive_val:.4f} ({sign}{diff_pp:.1f}pp) -> {winner}")

        # Speed comparison
        log(f"\n  Chunking time: {slumber_result.chunk_time:.1f}s vs {recursive_result.chunk_time:.1f}s")

    # Save results
    results_path = DATA_DIR / f"results_{args.chunk_size}T.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "corpus_size": len(corpus),
                "questions": len(questions),
                "chunk_size": args.chunk_size,
                "llm_model": args.llm_model,
                "embedding_model": args.model,
                "results": [
                    {
                        "chunker": r.chunker_name,
                        "num_chunks": r.num_chunks,
                        "avg_tokens": r.avg_tokens,
                        "hit_rates": r.hit_rates,
                        "mrr": r.mrr,
                        "chunk_time": r.chunk_time,
                        "hits": r.hits,
                    }
                    for r in results
                ],
            },
            f,
            indent=2,
        )
    log(f"\nResults saved to: {results_path}")

    log("\n" + "=" * 60)
    log("BENCHMARK COMPLETE")
    log("=" * 60)


if __name__ == "__main__":
    main()
