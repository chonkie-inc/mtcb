"""Cocha evaluator for Code Documents (Multi-language source code)."""

from typing import Any, Dict

from datasets import load_dataset

from ..benchmark import register_evaluator
from .base import BaseEvaluator


@register_evaluator("cocha")
class CochaEvaluator(BaseEvaluator):
    """Evaluator for the Cocha dataset (Code Documents).

    Cocha evaluates chunking performance on source code files from multiple
    programming languages. The dataset contains files from 10 languages:
    Python, JavaScript, TypeScript, Java, Go, Rust, C++, C, Ruby, and PHP.

    Each file is large enough to contain multiple functions/classes, making
    it suitable for evaluating code chunking strategies.

    This evaluator uses a global vector store where all chunks from all files
    are indexed together, simulating realistic RAG retrieval across a codebase.

    Example:
        >>> from chonkie import RecursiveChunker
        >>> evaluator = CochaEvaluator(
        ...     chunker=RecursiveChunker(chunk_size=512),
        ...     embedding_model="voyage-3-large",
        ...     cache_dir="./cache"
        ... )
        >>> result = evaluator.evaluate(k=[1, 3, 5, 10])
        >>> print(result)

    """

    DATASET_ID = "cocha"
    CORPUS_ITEM_NAME = "files"

    def _load_dataset(self) -> None:
        """Load the Cocha corpus and questions."""
        print("Loading Cocha datasets...")
        corpus_data = load_dataset("chonkie-ai/cocha", "corpus", split="train")
        questions_data = load_dataset("chonkie-ai/cocha", "questions", split="train")

        self.corpus = [doc.get("text", doc.get("content", "")) for doc in corpus_data]
        self.questions = [q["question"] for q in questions_data]
        self.relevant_passages = [q.get("chunk-must-contain", "") for q in questions_data]

        print(f"Loaded {len(self.corpus)} code files and {len(self.questions)} questions")

    def _get_extra_metadata(self) -> Dict[str, Any]:
        """Add language metadata for Cocha."""
        return {
            "languages": ["python", "javascript", "typescript", "java", "go", "rust", "c++", "c", "ruby", "php"],
        }
