"""Hojicha evaluator for Legal Contracts (CUAD derived)."""


from datasets import load_dataset

from ..benchmark import register_evaluator
from .base import BaseEvaluator


@register_evaluator("hojicha")
class HojichaEvaluator(BaseEvaluator):
    """Evaluator for the Hojicha dataset (Legal Contracts).

    Hojicha (HOldings JudIcial CHAllenges) evaluates chunking performance on
    legal contracts - testing how well chunkers handle formal legal language,
    nested clauses, cross-references, and structured contract sections.

    The dataset is derived from CUAD (Contract Understanding Atticus Dataset),
    containing 479 commercial contracts with 1,982 questions across 41 clause
    types (e.g., termination, liability caps, governing law).

    This evaluator uses a global vector store where all chunks from all documents
    are indexed together, simulating realistic RAG retrieval.

    Example:
        >>> from chonkie import RecursiveChunker
        >>> evaluator = HojichaEvaluator(
        ...     chunker=RecursiveChunker(chunk_size=1000),
        ...     embedding_model="voyage-3-large",
        ...     cache_dir="./cache"
        ... )
        >>> result = evaluator.evaluate(k=[1, 3, 5, 10])
        >>> print(result)

    """

    DATASET_ID = "hojicha"
    CORPUS_ITEM_NAME = "contracts"

    def _load_dataset(self) -> None:
        """Load the Hojicha corpus and questions."""
        print("Loading Hojicha datasets...")
        corpus_data = load_dataset("chonkie-ai/hojicha", "corpus", split="train")
        questions_data = load_dataset("chonkie-ai/hojicha", "questions", split="train")

        self.corpus = [doc["text"] for doc in corpus_data]
        self.questions = [q["question"] for q in questions_data]
        self.relevant_passages = [q["chunk-must-contain"] for q in questions_data]

        print(f"Loaded {len(self.corpus)} contracts and {len(self.questions)} questions")

    def _get_extra_metadata(self) -> dict:
        """Return extra metadata about the Hojicha dataset."""
        return {
            "source": "CUAD (Contract Understanding Atticus Dataset)",
            "domain": "legal_contracts",
            "question_types": 41,
        }
