"""Ryokucha evaluator for Medical Guidelines (NICE, CDC, WHO)."""


from datasets import load_dataset

from ..benchmark import register_evaluator
from .base import BaseEvaluator


@register_evaluator("ryokucha")
class RyokuchaEvaluator(BaseEvaluator):
    """Evaluator for the Ryokucha dataset (Medical Guidelines).

    Ryokucha (緑茶 - Green Tea, Medical Guidelines CHAllenges) evaluates
    chunking performance on clinical practice guidelines - testing how well
    chunkers handle medical terminology, structured recommendations, and
    evidence-based content.

    The dataset contains 241 authoritative clinical guidelines from NICE
    (UK), CDC (US), and WHO with 1,351 questions covering diagnoses,
    treatments, recommendations, and clinical procedures.

    This evaluator uses a global vector store where all chunks from all documents
    are indexed together, simulating realistic RAG retrieval.

    Example:
        >>> from chonkie import RecursiveChunker
        >>> evaluator = RyokuchaEvaluator(
        ...     chunker=RecursiveChunker(chunk_size=1000),
        ...     embedding_model="voyage-3-large",
        ...     cache_dir="./cache"
        ... )
        >>> result = evaluator.evaluate(k=[1, 3, 5, 10])
        >>> print(result)

    """

    DATASET_ID = "ryokucha"
    CORPUS_ITEM_NAME = "guidelines"

    def _load_dataset(self) -> None:
        """Load the Ryokucha corpus and questions."""
        print("Loading Ryokucha datasets...")
        corpus_data = load_dataset("chonkie-ai/ryokucha", "corpus", split="train")
        questions_data = load_dataset("chonkie-ai/ryokucha", "questions", split="train")

        self.corpus = [doc["text"] for doc in corpus_data]
        self.questions = [q["question"] for q in questions_data]
        self.relevant_passages = [q["chunk-must-contain"] for q in questions_data]

        print(f"Loaded {len(self.corpus)} guidelines and {len(self.questions)} questions")

    def _get_extra_metadata(self) -> dict:
        """Return extra metadata about the Ryokucha dataset."""
        return {
            "source": "NICE, CDC, WHO Clinical Guidelines",
            "domain": "medical_guidelines",
        }
