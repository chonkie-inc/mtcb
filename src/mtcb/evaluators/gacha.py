"""Gacha-specific evaluator implementation."""

import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from chonkie import Chunk
from ..cache import EvalCache
from ..embeddings import CatsuEmbeddings, get_embeddings, get_tokenizer_for_model, get_tokenizer_id_for_model
from ..evaluators.base import BaseEvaluator
from ..store import SimpleVectorStore
from ..types import EvalResult


def calculate_mrr(ranks: List[int]) -> float:
    """Calculate Mean Reciprocal Rank.

    Args:
        ranks: List of ranks (1-indexed, 0 means not found)

    Returns:
        MRR score

    """
    reciprocal_ranks = [1.0 / rank if rank > 0 else 0.0 for rank in ranks]
    return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0


class GachaEvaluator(BaseEvaluator):
    """Evaluator specifically designed for the Gacha dataset.

    This evaluator loads the Gacha dataset internally and evaluates retrieval performance
    across all books in the corpus. It computes recall and MRR scores across all books
    for given k values, providing a comprehensive evaluation of chunking and retrieval
    strategies on the Gacha benchmark.

    Args:
        chunker: The chunker to use for splitting documents
        embedding_model: The embedding model to use (default: "voyage-3-large")
        tokenizer: Tokenizer for chunking - "auto" to detect from model, or explicit identifier
        cache_dir: Directory for caching chunks and embeddings (None to disable)

    Example:
        >>> from chonkie import RecursiveChunker
        >>> evaluator = GachaEvaluator(
        ...     chunker=RecursiveChunker(chunk_size=1000),
        ...     embedding_model="voyage-3-large",
        ...     tokenizer="auto",
        ...     cache_dir="./cache"
        ... )
        >>> result = evaluator.evaluate(k=[1, 3, 5, 10])
        >>> print(result)

    """

    def __init__(
        self,
        chunker: Any,
        embedding_model: Union[str, Any] = "voyage-3-large",
        tokenizer: Optional[str] = "auto",
        cache_dir: Optional[str] = None,
    ) -> None:
        """Initialize the GachaEvaluator.

        Args:
            chunker: The chunker to use for splitting documents
            embedding_model: The embedding model to use. Can be a model name string
                (e.g., "voyage-3-large", "text-embedding-3-large") or an embeddings instance.
                For backward compatibility, "model2vec://" prefixed strings are also supported.
            tokenizer: Tokenizer for chunking. Use "auto" to auto-detect from embedding model,
                or provide an explicit tokenizer identifier (e.g., "voyageai/voyage-3-large").
                Set to None to skip tokenizer configuration.
            cache_dir: Directory for caching chunks and embeddings. Set to None to disable caching.

        """
        self.chunker = chunker

        # Set up embedding model
        if isinstance(embedding_model, str):
            self.embedding_model = get_embeddings(embedding_model)
            self._model_name = embedding_model
        else:
            self.embedding_model = embedding_model
            self._model_name = getattr(embedding_model, "model", str(embedding_model))

        # Set up tokenizer
        self._tokenizer = tokenizer
        self._tokenizer_id = get_tokenizer_id_for_model(self._model_name, tokenizer) if tokenizer else None

        # Set up caching
        self.cache = EvalCache(cache_dir)

        # Load datasets
        print("Loading Gacha datasets...")
        self.corpus = load_dataset("chonkie-ai/gacha", "corpus", split="train")
        self.questions = load_dataset("chonkie-ai/gacha", "questions", split="train")
        print(f"Loaded {len(self.corpus)} books and {len(self.questions)} questions")

        # Pre-compute question embeddings if caching is enabled
        self._question_embeddings: Optional[List[np.ndarray]] = None

    def _chunk(self, text: str) -> List[Chunk]:
        """Chunk a single text.

        Args:
            text: The text to chunk

        Returns:
            List of Chunk objects

        """
        return self.chunker(text)

    def _get_question_embeddings(self) -> List[np.ndarray]:
        """Get embeddings for all questions (with caching)."""
        if self._question_embeddings is not None:
            return self._question_embeddings

        def compute_fn():
            question_texts = [q["question"] for q in self.questions]
            return self.embedding_model.embed_batch(question_texts, input_type="query", show_progress=True)

        self._question_embeddings = self.cache.get_or_compute_question_embeddings(
            dataset_id="gacha",
            model=self._model_name,
            compute_fn=compute_fn,
        )
        return self._question_embeddings

    def _evaluate_book(
        self,
        book_idx: int,
        k_values: List[int],
        question_embeddings: List[np.ndarray],
    ) -> Dict:
        """Evaluate a single book for multiple k values efficiently.

        Args:
            book_idx: Index of the book in the corpus
            k_values: List of k values to evaluate
            question_embeddings: Pre-computed question embeddings

        Returns:
            Results for this book including per-k metrics and ranks

        """
        book = self.corpus[book_idx]
        book_title = book["title"]

        # Filter questions for this book
        book_question_indices = [
            i for i, q in enumerate(self.questions) if q["title"] == book_title
        ]

        if len(book_question_indices) == 0:
            return {
                "title": book_title,
                "results_by_k": {
                    k: {"correct_count": 0, "total_count": 0, "accuracy": 0.0}
                    for k in k_values
                },
                "ranks": [],
                "num_chunks": 0,
            }

        # Chunk the book
        chunks = self._chunk(book["text"])
        num_chunks = len(chunks)

        if num_chunks == 0:
            return {
                "title": book_title,
                "results_by_k": {
                    k: {"correct_count": 0, "total_count": len(book_question_indices), "accuracy": 0.0}
                    for k in k_values
                },
                "ranks": [0] * len(book_question_indices),
                "num_chunks": 0,
            }

        # Create vector store for this book
        svs = SimpleVectorStore()
        embeddings = self.embedding_model.embed_batch([chunk.text for chunk in chunks])
        svs.add_batch(embeddings, chunks)

        # Initialize counters for each k value
        correct_counts = {k: 0 for k in k_values}
        max_k = max(k_values)
        ranks = []

        # Evaluate questions for this book
        for q_idx in book_question_indices:
            question = self.questions[q_idx]
            relevant_passage = question["chunk-must-contain"]
            q_emb = question_embeddings[q_idx]

            results = svs.query(q_emb, k=max_k)

            # Find rank of first relevant chunk
            first_relevant_rank = 0
            for rank, (_, _, chunk) in enumerate(results, start=1):
                if relevant_passage in chunk.text:
                    first_relevant_rank = rank
                    break
            ranks.append(first_relevant_rank)

            # Check relevance for each k value
            for k_val in k_values:
                top_k_results = results[:k_val]
                if any(relevant_passage in chunk.text for _, _, chunk in top_k_results):
                    correct_counts[k_val] += 1

        # Format results for each k value
        results_by_k = {}
        for k_val in k_values:
            results_by_k[k_val] = {
                "correct_count": correct_counts[k_val],
                "total_count": len(book_question_indices),
                "accuracy": correct_counts[k_val] / len(book_question_indices),
            }

        return {
            "title": book_title,
            "results_by_k": results_by_k,
            "ranks": ranks,
            "num_chunks": num_chunks,
        }

    def evaluate(
        self,
        k: Union[int, List[int]] = 1,
        show_progress_bar: bool = True,
    ) -> EvalResult:
        """Evaluate retrieval performance across all books in the Gacha dataset.

        Args:
            k: Either a single k value or a list of k values to evaluate
            show_progress_bar: Whether to show progress bar

        Returns:
            EvalResult containing metrics (recall, MRR) and performance statistics

        """
        eval_start_time = time.time()
        k_values = [k] if isinstance(k, int) else k

        print(f"Evaluating with k values: {k_values}")
        if self._tokenizer_id:
            print(f"Using tokenizer: {self._tokenizer_id}")

        # Get question embeddings (cached if enabled)
        print("Loading question embeddings...")
        question_embeddings = self._get_question_embeddings()

        # Calculate total corpus size
        total_corpus_size_mb = sum(
            len(book["text"].encode("utf-8")) for book in self.corpus
        ) / (1024 * 1024)

        # Track stats across all books
        total_chunks_created = 0
        total_time_to_chunk = 0
        book_results = []
        all_ranks = []

        # Progress bar for books
        progress_bar = tqdm(
            range(len(self.corpus)),
            desc="Books",
            disable=not show_progress_bar,
        )

        for book_idx in progress_bar:
            chunk_start = time.time()
            book_result = self._evaluate_book(book_idx, k_values, question_embeddings)
            chunk_end = time.time()

            # Track stats
            total_chunks_created += book_result["num_chunks"]
            total_time_to_chunk += chunk_end - chunk_start
            book_results.append(book_result)
            all_ranks.extend(book_result["ranks"])

            # Update progress bar
            progress_bar.set_postfix({"book": book_result["title"][:20] + "..."})

        eval_end_time = time.time()
        total_evaluation_time = eval_end_time - eval_start_time
        chunk_speed_mb_per_sec = (
            total_corpus_size_mb / total_time_to_chunk if total_time_to_chunk > 0 else 0
        )

        # Aggregate results for each k value
        total_questions = len(self.questions)
        overall_correct = {}

        for k_val in k_values:
            total_correct = sum(
                book_result["results_by_k"][k_val]["correct_count"]
                for book_result in book_results
            )
            overall_correct[k_val] = total_correct

        questions_per_second = (
            total_questions / total_evaluation_time if total_evaluation_time > 0 else 0
        )

        # Calculate MRR for each k value
        mrr_by_k = {}
        for k_val in k_values:
            # For MRR@k, we cap ranks at k (ranks > k count as not found)
            capped_ranks = [r if r <= k_val else 0 for r in all_ranks]
            mrr_by_k[k_val] = calculate_mrr(capped_ranks)

        # Build metrics dictionary
        metrics = {
            "recall": {k_val: overall_correct[k_val] / total_questions for k_val in k_values},
            "mrr": mrr_by_k,
        }

        # Build metadata
        metadata = {
            "chunker_type": type(self.chunker).__name__,
            "chunker_config": repr(self.chunker),
            "embedding_model": str(self.embedding_model),
            "embedding_model_name": self._model_name,
            "tokenizer": self._tokenizer_id,
            "dataset": "gacha",
            "num_books": len(self.corpus),
            "total_questions": total_questions,
            "total_correct": overall_correct,
            "cache_enabled": self.cache.enabled,
        }

        return EvalResult(
            metrics=metrics,
            metadata=metadata,
            total_corpus_size_mb=total_corpus_size_mb,
            total_time_to_chunk=total_time_to_chunk,
            chunk_speed_mb_per_sec=chunk_speed_mb_per_sec,
            total_chunks_created=total_chunks_created,
            total_evaluation_time=total_evaluation_time,
            questions_per_second=questions_per_second,
        )
