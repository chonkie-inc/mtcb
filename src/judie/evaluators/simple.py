"""Simple evaluator implementation."""

import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
from tqdm import tqdm

from chonkie import Chunk
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


class SimpleEvaluator(BaseEvaluator):
    """Simple evaluator for retrieval-based question answering systems.

    This evaluator assesses the performance of a chunking and retrieval system by:
    1. Chunking a corpus of documents using a provided chunker
    2. Embedding the chunks using a specified embedding model
    3. Storing chunks and embeddings in a vector store
    4. For each question, retrieving the top-k most similar chunks
    5. Checking if any retrieved chunks contain the relevant passages
    6. Computing recall and MRR metrics based on successful retrievals

    The evaluator supports both single documents (strings) and collections of documents (lists).

    Args:
        corpus: The corpus to evaluate on (single string or list of strings)
        questions: List of questions to evaluate
        relevant_passages: The relevant passages for each question
        chunker: The chunker to use for splitting documents
        embedding_model: The embedding model to use (default: "voyage-3-large")
        tokenizer: Tokenizer - "auto" to detect from model, or explicit identifier
        cache_dir: Directory for caching (None to disable)

    Example:
        >>> evaluator = SimpleEvaluator(
        ...     corpus=["document1...", "document2..."],
        ...     questions=["What is X?", "How does Y work?"],
        ...     relevant_passages=["passage about X", "passage about Y"],
        ...     chunker=RecursiveChunker(chunk_size=1000),
        ...     embedding_model="voyage-3-large",
        ... )
        >>> result = evaluator.evaluate(k=[1, 3, 5, 10])
        >>> print(result)

    """

    def __init__(
        self,
        corpus: Union[str, List[str]],
        questions: List[str],
        relevant_passages: Union[str, List[str]],
        chunker: Any,
        embedding_model: Union[str, Any] = "voyage-3-large",
        tokenizer: Optional[str] = "auto",
        cache_dir: Optional[str] = None,
    ) -> None:
        """Initialize the SimpleEvaluator.

        Args:
            corpus: The corpus to evaluate on (single string or list of strings)
            questions: List of questions to evaluate
            relevant_passages: The relevant passages for each question
            chunker: The chunker to use for splitting documents
            embedding_model: The embedding model to use. Can be a model name string
                or an embeddings instance. For backward compatibility, "model2vec://"
                prefixed strings are also supported.
            tokenizer: Tokenizer for chunking. Use "auto" to auto-detect from model,
                or provide an explicit identifier. Set to None to skip.
            cache_dir: Directory for caching. Set to None to disable.

        """
        self.corpus = corpus
        self.questions = questions
        self.relevant_passages = relevant_passages
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

    def _chunk(self, corpus: Union[str, List[str]]) -> List[Chunk]:
        """Chunk the corpus.

        Args:
            corpus: The corpus to chunk

        Returns:
            List of Chunk objects

        """
        if isinstance(corpus, str):
            return self.chunker(corpus)
        else:
            chunks = []
            for text in corpus:
                chunks.extend(self.chunker(text))
            return chunks

    def evaluate(
        self,
        k: Union[int, List[int]] = 1,
        show_progress_bar: bool = True,
    ) -> EvalResult:
        """Evaluate the corpus.

        Args:
            k: Either a single k value or a list of k values to evaluate
            show_progress_bar: Whether to show progress bar

        Returns:
            EvalResult with recall, MRR metrics and performance stats

        """
        eval_start_time = time.time()

        # Normalize k to a list for uniform processing
        k_values = [k] if isinstance(k, int) else k
        max_k = max(k_values)

        # Calculate corpus size
        if isinstance(self.corpus, str):
            corpus_size_mb = len(self.corpus.encode("utf-8")) / (1024 * 1024)
        else:
            corpus_size_mb = sum(len(text.encode("utf-8")) for text in self.corpus) / (1024 * 1024)

        # Time the chunking process
        chunk_start_time = time.time()
        chunks = self._chunk(self.corpus)
        chunk_end_time = time.time()

        total_time_to_chunk = chunk_end_time - chunk_start_time
        chunk_speed_mb_per_sec = corpus_size_mb / total_time_to_chunk if total_time_to_chunk > 0 else 0

        # Create vector store and embed chunks
        svs = SimpleVectorStore()
        embeddings = self.embedding_model.embed_batch([chunk.text for chunk in chunks])
        svs.add_batch(embeddings, chunks)

        # Initialize results for each k value
        results_by_k = {k_val: 0 for k_val in k_values}
        ranks = []

        # Progress bar
        progress_bar = tqdm(
            enumerate(self.questions),
            total=len(self.questions),
            disable=not show_progress_bar,
        )

        # Evaluate each question
        for i, question in progress_bar:
            qemb = self.embedding_model.embed(question)
            results = svs.query(qemb, k=max_k)

            # Get the relevant passages for this specific question
            current_relevant_passages = self.relevant_passages[i]
            if isinstance(current_relevant_passages, str):
                current_relevant_passages = [current_relevant_passages]

            # Find rank of first relevant chunk
            first_relevant_rank = 0
            for rank, (_, _, chunk) in enumerate(results, start=1):
                if any(passage in chunk.text for passage in current_relevant_passages):
                    first_relevant_rank = rank
                    break
            ranks.append(first_relevant_rank)

            # Check for each k value
            for k_val in k_values:
                top_k_results = results[:k_val]
                if any(
                    any(passage in chunk.text for passage in current_relevant_passages)
                    for _, _, chunk in top_k_results
                ):
                    results_by_k[k_val] += 1

        eval_end_time = time.time()
        total_evaluation_time = eval_end_time - eval_start_time
        questions_per_second = len(self.questions) / total_evaluation_time if total_evaluation_time > 0 else 0

        # Calculate MRR for each k value
        mrr_by_k = {}
        for k_val in k_values:
            capped_ranks = [r if r <= k_val else 0 for r in ranks]
            mrr_by_k[k_val] = calculate_mrr(capped_ranks)

        # Build metrics dictionary
        metrics = {
            "recall": {k_val: results_by_k[k_val] / len(self.questions) for k_val in k_values},
            "mrr": mrr_by_k,
        }

        # Build metadata
        metadata = {
            "chunker_type": type(self.chunker).__name__,
            "chunker_config": repr(self.chunker),
            "embedding_model": str(self.embedding_model),
            "embedding_model_name": self._model_name,
            "tokenizer": self._tokenizer_id,
            "total_questions": len(self.questions),
            "total_correct": results_by_k,
            "cache_enabled": self.cache.enabled,
        }

        return EvalResult(
            metrics=metrics,
            metadata=metadata,
            total_corpus_size_mb=corpus_size_mb,
            total_time_to_chunk=total_time_to_chunk,
            chunk_speed_mb_per_sec=chunk_speed_mb_per_sec,
            total_chunks_created=len(chunks),
            total_evaluation_time=total_evaluation_time,
            questions_per_second=questions_per_second,
        )
