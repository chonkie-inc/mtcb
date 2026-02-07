"""Tacha evaluator for Financial Tables (TAT-QA derived)."""

from datasets import load_dataset

from ..benchmark import register_evaluator
from .base import BaseEvaluator


@register_evaluator("tacha")
class TachaEvaluator(BaseEvaluator):
    """Evaluator for the Tacha dataset (Financial Tables).

    Tacha evaluates chunking performance on financial documents containing
    tables and text - testing how well chunkers preserve table structure
    and keep headers with data rows.

    The dataset is derived from TAT-QA, containing financial reports with
    tables (12+ rows, 4+ columns) and associated paragraphs.

    This evaluator uses a global vector store where all chunks from all documents
    are indexed together, simulating realistic RAG retrieval.

    Example:
        >>> from chonkie import RecursiveChunker
        >>> evaluator = TachaEvaluator(
        ...     chunker=RecursiveChunker(chunk_size=1000),
        ...     embedding_model="voyage-3-large",
        ...     cache_dir="./cache"
        ... )
        >>> result = evaluator.evaluate(k=[1, 3, 5, 10])
        >>> print(result)

    """

    DATASET_ID = "tacha"
    CORPUS_ITEM_NAME = "documents"

    def _load_dataset(self) -> None:
        """Load the Tacha corpus and questions."""
        print("Loading Tacha datasets...")
        corpus_data = load_dataset("chonkie-ai/tacha", "corpus", split="train")
        questions_data = load_dataset("chonkie-ai/tacha", "questions", split="train")

        self.corpus = [doc["text"] for doc in corpus_data]
        self.questions = [q["question"] for q in questions_data]
        self.relevant_passages = [q["chunk-must-contain"] for q in questions_data]

        print(f"Loaded {len(self.corpus)} financial documents and {len(self.questions)} questions")

    def _get_extra_metadata(self) -> dict:
        """Return extra metadata about the Tacha dataset."""
        return {
            "source": "TAT-QA (filtered)",
            "domain": "financial_tables",
            "min_table_rows": 12,
            "min_table_cols": 4,
        }
