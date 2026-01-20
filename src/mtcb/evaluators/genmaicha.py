"""Genmaicha evaluator for MIT OCW Lecture Transcripts."""


from datasets import load_dataset

from ..benchmark import register_evaluator
from .base import BaseEvaluator


@register_evaluator("genmaicha")
class GenmaiichaEvaluator(BaseEvaluator):
    """Evaluator for the Genmaicha dataset (MIT OCW Lecture Transcripts).

    Genmaicha (玄米茶 - Brown Rice Tea, Educational Lecture CHAllenges) evaluates
    chunking performance on educational lecture transcripts - testing how well
    chunkers handle spoken language, technical explanations, and the natural
    flow of educational content.

    The dataset contains 250 MIT OpenCourseWare lecture transcripts across
    physics, biology, and chemistry courses with 2,193 questions covering
    concepts, definitions, derivations, and explanations from the lectures.

    This evaluator uses a global vector store where all chunks from all transcripts
    are indexed together, simulating realistic RAG retrieval.

    Example:
        >>> from chonkie import RecursiveChunker
        >>> evaluator = GenmaiichaEvaluator(
        ...     chunker=RecursiveChunker(chunk_size=1000),
        ...     embedding_model="voyage-3-large",
        ...     cache_dir="./cache"
        ... )
        >>> result = evaluator.evaluate(k=[1, 3, 5, 10])
        >>> print(result)

    """

    DATASET_ID = "genmaicha"
    CORPUS_ITEM_NAME = "transcripts"

    def _load_dataset(self) -> None:
        """Load the Genmaicha corpus and questions."""
        print("Loading Genmaicha datasets...")
        corpus_data = load_dataset("chonkie-ai/genmaicha", "corpus", split="train")
        questions_data = load_dataset("chonkie-ai/genmaicha", "questions", split="train")

        self.corpus = [doc["text"] for doc in corpus_data]
        self.questions = [q["question"] for q in questions_data]
        self.relevant_passages = [q["chunk-must-contain"] for q in questions_data]

        print(f"Loaded {len(self.corpus)} transcripts and {len(self.questions)} questions")

    def _get_extra_metadata(self) -> dict:
        """Return extra metadata about the Genmaicha dataset."""
        return {
            "source": "MIT OpenCourseWare",
            "domain": "educational_lectures",
            "topics": ["physics", "biology", "chemistry"],
        }
