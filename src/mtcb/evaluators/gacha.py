"""Gacha evaluator for Books/Literature corpus."""

from datasets import load_dataset

from ..benchmark import register_evaluator
from .base import BaseEvaluator


@register_evaluator("gacha")
class GachaEvaluator(BaseEvaluator):
    """Evaluator for the Gacha dataset (Books/Literature).

    Gacha evaluates chunking performance on full-length books and literature -
    long-form prose documents that test a chunker's ability to preserve narrative
    context and semantic boundaries.

    This evaluator uses a global vector store where all chunks from all books are
    indexed together, simulating realistic RAG retrieval across a document collection.

    Example:
        >>> from chonkie import RecursiveChunker
        >>> evaluator = GachaEvaluator(
        ...     chunker=RecursiveChunker(chunk_size=1000),
        ...     embedding_model="voyage-3-large",
        ...     cache_dir="./cache"
        ... )
        >>> result = evaluator.evaluate(k=[1, 3, 5, 10])
        >>> print(result)

    """

    DATASET_ID = "gacha"
    CORPUS_ITEM_NAME = "books"

    def _load_dataset(self) -> None:
        """Load the Gacha corpus and questions."""
        print("Loading Gacha datasets...")
        corpus_data = load_dataset("chonkie-ai/gacha", "corpus", split="train")
        questions_data = load_dataset("chonkie-ai/gacha", "questions", split="train")

        self.corpus = [doc["text"] for doc in corpus_data]
        self.questions = [q["question"] for q in questions_data]
        self.relevant_passages = [q["chunk-must-contain"] for q in questions_data]

        print(f"Loaded {len(self.corpus)} books and {len(self.questions)} questions")
