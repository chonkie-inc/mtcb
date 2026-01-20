"""Ficha evaluator for Financial Documents (SEC 10-K/10-Q filings)."""


from datasets import load_dataset

from ..benchmark import register_evaluator
from .base import BaseEvaluator


@register_evaluator("ficha")
class FichaEvaluator(BaseEvaluator):
    """Evaluator for the Ficha dataset (Financial Documents).

    Ficha evaluates chunking performance on SEC 10-K and 10-Q filings -
    complex financial documents with tables, numeric data, and regulatory text.

    The dataset contains Fortune 100 company annual reports with questions
    about revenue, risk factors, financial metrics, and management outlook.

    This evaluator uses a global vector store where all chunks from all filings
    are indexed together, simulating realistic RAG retrieval across a document collection.

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

    DATASET_ID = "ficha"
    CORPUS_ITEM_NAME = "filings"

    def _load_dataset(self) -> None:
        """Load the Ficha corpus and questions."""
        print("Loading Ficha datasets...")
        corpus_data = load_dataset("chonkie-ai/ficha", "corpus", split="train")
        questions_data = load_dataset("chonkie-ai/ficha", "questions", split="train")

        self.corpus = [doc.get("text", doc.get("content", "")) for doc in corpus_data]
        self.questions = [q["question"] for q in questions_data]
        self.relevant_passages = [
            q.get("supporting_passage", q.get("chunk-must-contain", ""))
            for q in questions_data
        ]

        print(f"Loaded {len(self.corpus)} filings and {len(self.questions)} questions")
