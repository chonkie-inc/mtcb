"""Simple evaluator implementation."""

from typing import Any, List, Optional, Union

from .base import BaseEvaluator


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

    DATASET_ID = "simple"
    CORPUS_ITEM_NAME = "documents"

    def __init__(
        self,
        corpus: Union[str, List[str]],
        questions: List[str],
        relevant_passages: List[str],
        chunker: Any,
        embedding_model: Union[str, Any] = "voyage-3-large",
        tokenizer: Optional[str] = "auto",
        cache_dir: Optional[str] = None,
        batch_size: int = 128,
    ) -> None:
        """Initialize the SimpleEvaluator.

        Args:
            corpus: The corpus to evaluate on (single string or list of strings)
            questions: List of questions to evaluate
            relevant_passages: The relevant passages for each question
            chunker: The chunker to use for splitting documents
            embedding_model: The embedding model to use. Can be a model name string
                or an embeddings instance.
            tokenizer: Tokenizer for chunking. Use "auto" to auto-detect from model,
                or provide an explicit identifier. Set to None to skip.
            cache_dir: Directory for caching. Set to None to disable.
            batch_size: Batch size for embedding API calls. Default 128.

        """
        # Store init args for _load_dataset to use
        self._init_corpus = corpus
        self._init_questions = questions
        self._init_relevant_passages = relevant_passages

        # Call parent init (which calls _load_dataset)
        super().__init__(
            chunker=chunker,
            embedding_model=embedding_model,
            tokenizer=tokenizer,
            cache_dir=cache_dir,
            batch_size=batch_size,
        )

    def _load_dataset(self) -> None:
        """Load data from constructor arguments."""
        # Normalize corpus to list
        if isinstance(self._init_corpus, str):
            self.corpus = [self._init_corpus]
        else:
            self.corpus = list(self._init_corpus)

        self.questions = list(self._init_questions)
        self.relevant_passages = list(self._init_relevant_passages)

        # Clean up init args
        del self._init_corpus
        del self._init_questions
        del self._init_relevant_passages
