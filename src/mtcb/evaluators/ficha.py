"""Ficha evaluator for Financial Documents (SEC 10-K/10-Q filings)."""

import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from chonkie import Chunk
from ..benchmark import register_evaluator
from ..cache import EvalCache
from ..embeddings import get_embeddings, get_tokenizer_id_for_model
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


@register_evaluator("ficha")
class FichaEvaluator(BaseEvaluator):
    """Evaluator for the Ficha dataset (Financial Documents).

    Ficha evaluates chunking performance on SEC 10-K and 10-Q filings -
    complex financial documents with tables, numeric data, and regulatory text.

    The dataset contains Fortune 100 company annual reports with questions
    about revenue, risk factors, financial metrics, and management outlook.

    Args:
        chunker: The chunker to use for splitting documents
        embedding_model: The embedding model to use (default: "voyage-3-large")
        tokenizer: Tokenizer for chunking - "auto" to detect from model
        cache_dir: Directory for caching chunks and embeddings (None to disable)

    Example:
        >>> from chonkie import RecursiveChunker
        >>> evaluator = FichaEvaluator(
        ...     chunker=RecursiveChunker(chunk_size=1000),
        ...     embedding_model="voyage-3-large",
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
        """Initialize the FichaEvaluator."""
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
        print("Loading Ficha datasets...")
        self.corpus = load_dataset("chonkie-ai/ficha", "corpus", split="train")
        self.questions = load_dataset("chonkie-ai/ficha", "questions", split="train")
        print(f"Loaded {len(self.corpus)} filings and {len(self.questions)} questions")

        # Pre-compute question embeddings
        self._question_embeddings: Optional[List[np.ndarray]] = None

    def _chunk(self, text: str) -> List[Chunk]:
        """Chunk a single text."""
        return self.chunker(text)

    def _get_question_embeddings(self) -> List[np.ndarray]:
        """Get embeddings for all questions (with caching)."""
        if self._question_embeddings is not None:
            return self._question_embeddings

        def compute_fn():
            question_texts = [q["question"] for q in self.questions]
            return self.embedding_model.embed_batch(question_texts, input_type="query", show_progress=True)

        self._question_embeddings = self.cache.get_or_compute_question_embeddings(
            dataset_id="ficha",
            model=self._model_name,
            compute_fn=compute_fn,
        )
        return self._question_embeddings

    def _evaluate_filing(
        self,
        filing_idx: int,
        k_values: List[int],
        question_embeddings: List[np.ndarray],
    ) -> Dict:
        """Evaluate a single filing for multiple k values."""
        filing = self.corpus[filing_idx]
        # Use ticker as the identifier (matches Ficha dataset structure)
        ticker = filing.get("ticker", filing.get("id", f"filing_{filing_idx}"))

        # Filter questions for this filing
        filing_question_indices = [
            i for i, q in enumerate(self.questions)
            if q.get("ticker", q.get("id")) == ticker
        ]

        if len(filing_question_indices) == 0:
            return {
                "ticker": ticker,
                "results_by_k": {
                    k: {"correct_count": 0, "total_count": 0, "accuracy": 0.0}
                    for k in k_values
                },
                "ranks": [],
                "num_chunks": 0,
            }

        # Chunk the filing
        text = filing.get("text", filing.get("content", ""))
        chunks = self._chunk(text)
        num_chunks = len(chunks)

        if num_chunks == 0:
            return {
                "ticker": ticker,
                "results_by_k": {
                    k: {"correct_count": 0, "total_count": len(filing_question_indices), "accuracy": 0.0}
                    for k in k_values
                },
                "ranks": [0] * len(filing_question_indices),
                "num_chunks": 0,
            }

        # Create vector store
        svs = SimpleVectorStore()
        embeddings = self.embedding_model.embed_batch([chunk.text for chunk in chunks])
        svs.add_batch(embeddings, chunks)

        # Initialize counters
        correct_counts = {k: 0 for k in k_values}
        max_k = max(k_values)
        ranks = []

        # Evaluate questions
        for q_idx in filing_question_indices:
            question = self.questions[q_idx]
            # Ficha uses "supporting_passage" field
            relevant_passage = question.get("supporting_passage", question.get("chunk-must-contain", ""))
            q_emb = question_embeddings[q_idx]

            results = svs.query(q_emb, k=max_k)

            # Find rank of first relevant chunk
            first_relevant_rank = 0
            for rank, (_, _, chunk) in enumerate(results, start=1):
                if relevant_passage in chunk.text:
                    first_relevant_rank = rank
                    break
            ranks.append(first_relevant_rank)

            # Check relevance for each k
            for k_val in k_values:
                top_k_results = results[:k_val]
                if any(relevant_passage in chunk.text for _, _, chunk in top_k_results):
                    correct_counts[k_val] += 1

        # Format results
        results_by_k = {}
        for k_val in k_values:
            results_by_k[k_val] = {
                "correct_count": correct_counts[k_val],
                "total_count": len(filing_question_indices),
                "accuracy": correct_counts[k_val] / len(filing_question_indices),
            }

        return {
            "ticker": ticker,
            "results_by_k": results_by_k,
            "ranks": ranks,
            "num_chunks": num_chunks,
        }

    def evaluate(
        self,
        k: Union[int, List[int]] = 1,
        show_progress_bar: bool = True,
    ) -> EvalResult:
        """Evaluate retrieval performance across all filings in Ficha.

        Args:
            k: Either a single k value or a list of k values
            show_progress_bar: Whether to show progress bar

        Returns:
            EvalResult containing metrics and performance statistics
        """
        eval_start_time = time.time()
        k_values = [k] if isinstance(k, int) else k

        print(f"Evaluating with k values: {k_values}")
        if self._tokenizer_id:
            print(f"Using tokenizer: {self._tokenizer_id}")

        # Get question embeddings
        print("Loading question embeddings...")
        question_embeddings = self._get_question_embeddings()

        # Calculate corpus size
        total_corpus_size_mb = sum(
            len(filing.get("text", filing.get("content", "")).encode("utf-8"))
            for filing in self.corpus
        ) / (1024 * 1024)

        # Track stats
        total_chunks_created = 0
        total_time_to_chunk = 0
        filing_results = []
        all_ranks = []

        # Progress bar
        progress_bar = tqdm(
            range(len(self.corpus)),
            desc="Filings",
            disable=not show_progress_bar,
        )

        for filing_idx in progress_bar:
            chunk_start = time.time()
            filing_result = self._evaluate_filing(filing_idx, k_values, question_embeddings)
            chunk_end = time.time()

            total_chunks_created += filing_result["num_chunks"]
            total_time_to_chunk += chunk_end - chunk_start
            filing_results.append(filing_result)
            all_ranks.extend(filing_result["ranks"])

            progress_bar.set_postfix({"ticker": str(filing_result["ticker"])})

        eval_end_time = time.time()
        total_evaluation_time = eval_end_time - eval_start_time
        chunk_speed_mb_per_sec = (
            total_corpus_size_mb / total_time_to_chunk if total_time_to_chunk > 0 else 0
        )

        # Aggregate results
        total_questions = len(self.questions)
        overall_correct = {}

        for k_val in k_values:
            total_correct = sum(
                filing_result["results_by_k"][k_val]["correct_count"]
                for filing_result in filing_results
            )
            overall_correct[k_val] = total_correct

        questions_per_second = (
            total_questions / total_evaluation_time if total_evaluation_time > 0 else 0
        )

        # Calculate MRR
        mrr_by_k = {}
        for k_val in k_values:
            capped_ranks = [r if r <= k_val else 0 for r in all_ranks]
            mrr_by_k[k_val] = calculate_mrr(capped_ranks)

        # Build metrics
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
            "dataset": "ficha",
            "num_filings": len(self.corpus),
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
