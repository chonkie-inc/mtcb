"""Sencha evaluator for Scientific Papers (NLP papers from QASPER)."""


from datasets import load_dataset

from ..benchmark import register_evaluator
from .base import BaseEvaluator


@register_evaluator("sencha")
class SenchaEvaluator(BaseEvaluator):
    """Evaluator for the Sencha dataset (Scientific Papers).

    Sencha (Scientific ENterprise CHAllenges) evaluates chunking performance on
    academic NLP papers - testing how well chunkers handle structured scientific
    documents with sections, citations, and technical content.

    The dataset is derived from QASPER, containing full-text NLP papers with
    questions that require understanding specific passages.

    This evaluator uses a global vector store where all chunks from all documents
    are indexed together, simulating realistic RAG retrieval.

    Example:
        >>> from chonkie import RecursiveChunker
        >>> evaluator = SenchaEvaluator(
        ...     chunker=RecursiveChunker(chunk_size=1000),
        ...     embedding_model="voyage-3-large",
        ...     cache_dir="./cache"
        ... )
        >>> result = evaluator.evaluate(k=[1, 3, 5, 10])
        >>> print(result)

    """

    DATASET_ID = "sencha"
    CORPUS_ITEM_NAME = "papers"

    def _load_dataset(self) -> None:
        """Load the Sencha corpus and questions."""
        print("Loading Sencha datasets...")
        corpus_data = load_dataset("chonkie-ai/sencha", "corpus", split="train")
        questions_data = load_dataset("chonkie-ai/sencha", "questions", split="train")

        self.corpus = [doc["text"] for doc in corpus_data]
        self.questions = [q["question"] for q in questions_data]
        self.relevant_passages = [q["chunk-must-contain"] for q in questions_data]

        print(f"Loaded {len(self.corpus)} papers and {len(self.questions)} questions")

    def _get_extra_metadata(self) -> dict:
        """Return extra metadata about the Sencha dataset."""
        return {
            "source": "QASPER",
            "domain": "nlp_papers",
        }
