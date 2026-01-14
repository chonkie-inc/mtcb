#!/usr/bin/env python3
"""Chunk the entire Gacha corpus with NeuralChunker.

Saves progress incrementally to allow resuming if interrupted.

Usage:
    python benchmark_neural.py
    python benchmark_neural.py --model mirth/chonky_modernbert_base_1
    python benchmark_neural.py --resume
"""

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

from chonkie import NeuralChunker
from tokenizers import Tokenizer

DATA_DIR = Path(__file__).parent / "data" / "gacha_benchmark"
DEFAULT_OUTPUT = Path(__file__).parent / "data" / "gacha_chunks_neural.jsonl"

# Available models
MODELS = {
    "distilbert": "mirth/chonky_distilbert_base_uncased_1",
    "modernbert-base": "mirth/chonky_modernbert_base_1",
    "modernbert-large": "mirth/chonky_modernbert_large_1",
    "multilingual": "mirth/chonky_mmbert_small_multilingual_1",
}

DEFAULT_MODEL = "distilbert"


@dataclass
class Chunk:
    text: str
    start_index: int
    end_index: int
    token_count: int


@dataclass
class BookChunks:
    book_index: int
    title: str
    num_chunks: int
    chunks: List[dict]
    processing_time: float
    model: str


def load_processed_books(output_path: Path) -> set[int]:
    """Load set of already processed book indices."""
    processed = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed.add(data["book_index"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return processed


def save_book_chunks(output_path: Path, book_chunks: BookChunks):
    """Append book chunks to output file."""
    with open(output_path, "a") as f:
        f.write(json.dumps(asdict(book_chunks)) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Chunk Gacha corpus with NeuralChunker")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSONL file")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        choices=list(MODELS.keys()), help="Model to use")
    parser.add_argument("--start", type=int, default=0, help="Start from book index")
    parser.add_argument("--end", type=int, default=None, help="End at book index (exclusive)")
    args = parser.parse_args()

    # Load corpus
    print("Loading corpus...")
    with open(DATA_DIR / "gacha_corpus.jsonl") as f:
        corpus = [json.loads(line) for line in f]

    print(f"Total books: {len(corpus)}")

    # Check for resume
    processed = set()
    if args.resume and args.output.exists():
        processed = load_processed_books(args.output)
        print(f"Resuming: {len(processed)} books already processed")
    elif not args.resume and args.output.exists():
        # Backup existing file
        backup_path = args.output.with_suffix(".jsonl.bak")
        args.output.rename(backup_path)
        print(f"Backed up existing output to {backup_path}")

    # Initialize
    model_name = MODELS[args.model]
    print(f"\nLoading NeuralChunker model: {model_name}")
    chunker = NeuralChunker(model=model_name)

    # Tokenizer for counting tokens
    tokenizer = Tokenizer.from_pretrained("voyageai/voyage-3-large")

    # Determine range
    start_idx = args.start
    end_idx = args.end if args.end is not None else len(corpus)

    # Process books
    total_chunks = 0
    total_time = 0

    print(f"\nProcessing books {start_idx} to {end_idx - 1}...")
    print(f"Model: {args.model} ({model_name})")
    print("=" * 70)

    for book_idx in range(start_idx, end_idx):
        if book_idx in processed:
            print(f"[{book_idx:3d}] {corpus[book_idx]['title'][:40]:<40} SKIPPED (already processed)")
            continue

        book = corpus[book_idx]
        title = book["title"][:40]

        print(f"[{book_idx:3d}] {title:<40}", end=" ", flush=True)

        try:
            start = time.time()
            raw_chunks = chunker.chunk(book["text"])
            elapsed = time.time() - start

            # Convert to our format with token counts
            chunks = []
            for c in raw_chunks:
                token_count = len(tokenizer.encode(c.text).ids)
                chunks.append(Chunk(
                    text=c.text,
                    start_index=c.start_index,
                    end_index=c.end_index,
                    token_count=token_count,
                ))

            # Save immediately
            book_chunks = BookChunks(
                book_index=book_idx,
                title=book["title"],
                num_chunks=len(chunks),
                chunks=[asdict(c) for c in chunks],
                processing_time=elapsed,
                model=args.model,
            )
            save_book_chunks(args.output, book_chunks)

            total_chunks += len(chunks)
            total_time += elapsed

            chars_per_sec = len(book["text"]) / elapsed
            mean_tokens = sum(c.token_count for c in chunks) / len(chunks) if chunks else 0
            print(f"chunks={len(chunks):>4}, time={elapsed:>5.1f}s, {chars_per_sec:>6.0f} char/s, mean_tok={mean_tokens:.0f}")

        except Exception as e:
            print(f"ERROR: {e}")
            continue

    # Summary
    print("=" * 70)
    print(f"COMPLETE: {end_idx - start_idx - len(processed)} books processed")
    print(f"Total chunks: {total_chunks}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()
