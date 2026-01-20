"""Macha evaluator for Technical Documentation (GitHub READMEs)."""


from datasets import load_dataset

from ..benchmark import register_evaluator
from .base import BaseEvaluator


@register_evaluator("macha")
class MachaEvaluator(BaseEvaluator):
    """Evaluator for the Macha dataset (Technical Documentation).

    Macha evaluates chunking performance on GitHub README files - semi-structured
    markdown documents commonly used in developer Q&A and documentation retrieval.

    The dataset contains README files from popular GitHub repositories with
    questions about installation, usage, configuration, and API details.

    This evaluator uses a global vector store where all chunks from all documents
    are indexed together, simulating realistic RAG retrieval across a documentation collection.

    Example:
        >>> from chonkie import RecursiveChunker
        >>> evaluator = MachaEvaluator(
        ...     chunker=RecursiveChunker(chunk_size=1000),
        ...     embedding_model="voyage-3-large",
        ...     cache_dir="./cache"
        ... )
        >>> result = evaluator.evaluate(k=[1, 3, 5, 10])
        >>> print(result)

    """

    DATASET_ID = "macha"
    CORPUS_ITEM_NAME = "documents"

    def _load_dataset(self) -> None:
        """Load the Macha corpus and questions."""
        print("Loading Macha datasets...")
        corpus_data = load_dataset("chonkie-ai/macha", "corpus", split="train")
        questions_data = load_dataset("chonkie-ai/macha", "questions", split="train")

        self.corpus = [doc.get("text", doc.get("content", "")) for doc in corpus_data]
        self.questions = [q["question"] for q in questions_data]
        self.relevant_passages = [
            q.get("chunk-must-contain", q.get("supporting_passage", ""))
            for q in questions_data
        ]

        print(f"Loaded {len(self.corpus)} READMEs and {len(self.questions)} questions")
