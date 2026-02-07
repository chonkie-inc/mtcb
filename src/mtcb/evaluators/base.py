"""Base class for all evaluators."""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
from chonkie import Chunk
from tqdm import tqdm

from ..cache import EvalCache
from ..embeddings import get_embeddings, get_tokenizer_id_for_model
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


def calculate_precision(ranks: List[int], k: int) -> float:
    """Calculate Precision@k.

    For single-relevant-item retrieval, precision@k is the fraction of
    queries where the relevant item appears in top-k, divided by k.

    Args:
        ranks: List of ranks (1-indexed, 0 means not found)
        k: The cutoff value

    Returns:
        Precision@k score

    """
    # For each query: 1/k if relevant item in top-k, else 0
    precision_scores = [1.0 / k if 0 < rank <= k else 0.0 for rank in ranks]
    return float(np.mean(precision_scores)) if precision_scores else 0.0


def calculate_ndcg(ranks: List[int], k: int) -> float:
    """Calculate NDCG@k (Normalized Discounted Cumulative Gain).

    For binary relevance with a single relevant item per query:
    - DCG@k = 1/log2(rank+1) if rank <= k, else 0
    - IDCG@k = 1/log2(2) = 1 (ideal: relevant item at rank 1)
    - NDCG@k = DCG@k / IDCG@k = 1/log2(rank+1) if rank <= k

    Args:
        ranks: List of ranks (1-indexed, 0 means not found)
        k: The cutoff value

    Returns:
        NDCG@k score

    """
    ndcg_scores = []
    for rank in ranks:
        if 0 < rank <= k:
            # DCG = 1/log2(rank+1), IDCG = 1, so NDCG = 1/log2(rank+1)
            ndcg_scores.append(1.0 / np.log2(rank + 1))
        else:
            ndcg_scores.append(0.0)
    return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0


class BaseEvaluator(ABC):
    """Base class for all evaluators with full evaluation pipeline.

    Subclasses must:
    - Set DATASET_ID (short identifier like "gacha")
    - Set CORPUS_ITEM_NAME (plural name like "books", "filings")
    - Implement _load_dataset() to populate self.corpus, self.questions, self.relevant_passages

    Optionally override:
    - _get_extra_metadata(): Add dataset-specific metadata
    """

    # Class attributes - override in subclasses
    DATASET_ID: str = ""
    CORPUS_ITEM_NAME: str = ""

    def __init__(
        self,
        chunker: Any,
        embedding_model: Union[str, Any] = "voyage-3-large",
        tokenizer: Optional[str] = "auto",
        cache_dir: Optional[str] = None,
        batch_size: int = 128,
    ) -> None:
        """Initialize the evaluator.

        Args:
            chunker: The chunker to use for splitting documents
            embedding_model: The embedding model to use. Can be a model name string
                (e.g., "voyage-3-large", "text-embedding-3-large") or an embeddings instance.
            tokenizer: Tokenizer for chunking. Use "auto" to auto-detect from embedding model,
                or provide an explicit tokenizer identifier. Set to None to skip.
            cache_dir: Directory for caching chunks and embeddings. Set to None to disable.
            batch_size: Batch size for embedding API calls. Default 128.

        """
        self.chunker = chunker
        self.batch_size = batch_size

        # Set up embedding model
        if isinstance(embedding_model, str):
            self.embedding_model = get_embeddings(embedding_model, batch_size=batch_size)
            self._model_name = embedding_model
        else:
            self.embedding_model = embedding_model
            self._model_name = getattr(embedding_model, "model", str(embedding_model))

        # Set up tokenizer
        self._tokenizer = tokenizer
        self._tokenizer_id = (
            get_tokenizer_id_for_model(self._model_name, tokenizer) if tokenizer else None
        )

        # Set up caching
        self.cache = EvalCache(cache_dir)

        # Initialize data lists (to be populated by _load_dataset)
        self.corpus: List[str] = []
        self.questions: List[str] = []
        self.relevant_passages: List[str] = []

        # Load dataset (calls subclass implementation)
        self._load_dataset()

        # Pre-compute question embeddings placeholder
        self._question_embeddings: Optional[List[np.ndarray]] = None

    @abstractmethod
    def _load_dataset(self) -> None:
        """Load data and populate self.corpus, self.questions, self.relevant_passages."""
        pass

    def _get_extra_metadata(self) -> Dict[str, Any]:
        """Override to add dataset-specific metadata.

        Returns:
            Dictionary of extra metadata fields

        """
        return {}

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
            return self.embedding_model.embed_batch(
                self.questions, input_type="query", show_progress=True
            )

        self._question_embeddings = self.cache.get_or_compute_question_embeddings(
            dataset_id=self.DATASET_ID,
            model=self._model_name,
            compute_fn=compute_fn,
        )
        return self._question_embeddings

    def evaluate(
        self,
        k: Union[int, List[int]] = 1,
        show_progress_bar: bool = True,
    ) -> EvalResult:
        """Evaluate retrieval performance across all documents.

        Uses a global vector store where all chunks from all documents are indexed
        together, simulating realistic RAG retrieval.

        Args:
            k: Either a single k value or a list of k values to evaluate
            show_progress_bar: Whether to show progress bar

        Returns:
            EvalResult containing metrics (recall, MRR) and performance statistics

        """
        eval_start_time = time.time()
        k_values = [k] if isinstance(k, int) else k
        max_k = max(k_values)

        print(f"Evaluating with k values: {k_values}")
        if self._tokenizer_id:
            print(f"Using tokenizer: {self._tokenizer_id}")

        # Get question embeddings (cached if enabled)
        print("Loading question embeddings...")
        question_embeddings = self._get_question_embeddings()

        # Calculate total corpus size
        total_corpus_size_mb = sum(len(text.encode("utf-8")) for text in self.corpus) / (
            1024 * 1024
        )

        # Phase 1: Chunk all documents
        print("Chunking all documents...")
        chunk_start_time = time.time()
        all_chunks: List[Chunk] = []

        chunk_progress = tqdm(
            self.corpus,
            desc="Chunking",
            disable=not show_progress_bar,
        )
        for text in chunk_progress:
            chunks = self._chunk(text)
            all_chunks.extend(chunks)
            chunk_progress.set_postfix({"total_chunks": len(all_chunks)})

        chunk_end_time = time.time()
        total_time_to_chunk = chunk_end_time - chunk_start_time
        total_chunks_created = len(all_chunks)

        print(f"Created {total_chunks_created} chunks in {total_time_to_chunk:.2f}s")

        # Phase 2: Embed all chunks in batches
        print(f"Embedding all chunks (batch_size={self.batch_size})...")
        chunk_texts = [chunk.text for chunk in all_chunks]
        chunk_embeddings = self.embedding_model.embed_batch(chunk_texts, show_progress=True)

        # Phase 3: Build global vector store
        print("Building global vector store...")
        svs = SimpleVectorStore()
        svs.add_batch(chunk_embeddings, all_chunks)

        # Phase 4: Evaluate all questions
        print("Evaluating questions...")
        correct_counts = {k_val: 0 for k_val in k_values}
        all_ranks = []

        question_progress = tqdm(
            enumerate(self.questions),
            total=len(self.questions),
            desc="Questions",
            disable=not show_progress_bar,
        )

        for q_idx, question in question_progress:
            relevant_passage = self.relevant_passages[q_idx]
            q_emb = question_embeddings[q_idx]

            results = svs.query(q_emb, k=max_k)

            # Find rank of first relevant chunk
            first_relevant_rank = 0
            for rank, (_, _, chunk) in enumerate(results, start=1):
                if relevant_passage in chunk.text:
                    first_relevant_rank = rank
                    break
            all_ranks.append(first_relevant_rank)

            # Check relevance for each k value
            for k_val in k_values:
                top_k_results = results[:k_val]
                if any(relevant_passage in chunk.text for _, _, chunk in top_k_results):
                    correct_counts[k_val] += 1

        eval_end_time = time.time()
        total_evaluation_time = eval_end_time - eval_start_time
        chunk_speed_mb_per_sec = (
            total_corpus_size_mb / total_time_to_chunk if total_time_to_chunk > 0 else 0
        )

        total_questions = len(self.questions)
        questions_per_second = (
            total_questions / total_evaluation_time if total_evaluation_time > 0 else 0
        )

        # Calculate MRR, Precision, and NDCG for each k value
        mrr_by_k = {}
        precision_by_k = {}
        ndcg_by_k = {}
        for k_val in k_values:
            # For MRR@k, we cap ranks at k (ranks > k count as not found)
            capped_ranks = [r if r <= k_val else 0 for r in all_ranks]
            mrr_by_k[k_val] = calculate_mrr(capped_ranks)
            precision_by_k[k_val] = calculate_precision(all_ranks, k_val)
            ndcg_by_k[k_val] = calculate_ndcg(all_ranks, k_val)

        # Build metrics dictionary
        metrics = {
            "recall": {k_val: correct_counts[k_val] / total_questions for k_val in k_values},
            "precision": precision_by_k,
            "mrr": mrr_by_k,
            "ndcg": ndcg_by_k,
        }

        # Build metadata
        metadata = {
            "chunker_type": type(self.chunker).__name__,
            "chunker_config": repr(self.chunker),
            "embedding_model": str(self.embedding_model),
            "embedding_model_name": self._model_name,
            "tokenizer": self._tokenizer_id,
            "dataset": self.DATASET_ID,
            f"num_{self.CORPUS_ITEM_NAME}": len(self.corpus),
            "total_questions": total_questions,
            "total_correct": correct_counts,
            "cache_enabled": self.cache.enabled,
        }

        # Add any extra metadata from subclass
        metadata.update(self._get_extra_metadata())

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
