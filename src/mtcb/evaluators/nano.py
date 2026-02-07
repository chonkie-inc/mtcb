"""Nano evaluators for fast iteration and testing.

These evaluators use the nano datasets (~100 questions each) for quick
benchmarking during development and testing.
"""

from typing import Any, Dict

from datasets import load_dataset

from ..benchmark import register_evaluator
from .base import BaseEvaluator


@register_evaluator("nano-gacha")
class NanoGachaEvaluator(BaseEvaluator):
    """Nano evaluator for the Gacha dataset (Books/Literature).

    Uses a subset of ~100 questions for fast iteration.
    """

    DATASET_ID = "nano-gacha"
    CORPUS_ITEM_NAME = "books"

    def _load_dataset(self) -> None:
        """Load the nano Gacha corpus and questions."""
        print("Loading Nano Gacha datasets...")
        corpus_data = load_dataset("chonkie-ai/nano-gacha", "corpus", split="train")
        questions_data = load_dataset("chonkie-ai/nano-gacha", "questions", split="train")

        self.corpus = [doc["text"] for doc in corpus_data]
        self.questions = [q["question"] for q in questions_data]
        self.relevant_passages = [q["chunk-must-contain"] for q in questions_data]

        print(f"Loaded {len(self.corpus)} books and {len(self.questions)} questions")


@register_evaluator("nano-ficha")
class NanoFichaEvaluator(BaseEvaluator):
    """Nano evaluator for the Ficha dataset (Financial Documents).

    Uses a subset of ~100 questions for fast iteration.
    """

    DATASET_ID = "nano-ficha"
    CORPUS_ITEM_NAME = "filings"

    def _load_dataset(self) -> None:
        """Load the nano Ficha corpus and questions."""
        print("Loading Nano Ficha datasets...")
        corpus_data = load_dataset("chonkie-ai/nano-ficha", "corpus", split="train")
        questions_data = load_dataset("chonkie-ai/nano-ficha", "questions", split="train")

        self.corpus = [doc["text"] for doc in corpus_data]
        self.questions = [q["question"] for q in questions_data]
        self.relevant_passages = [q["chunk-must-contain"] for q in questions_data]

        print(f"Loaded {len(self.corpus)} filings and {len(self.questions)} questions")


@register_evaluator("nano-macha")
class NanoMachaEvaluator(BaseEvaluator):
    """Nano evaluator for the Macha dataset (Technical Documentation).

    Uses a subset of ~100 questions for fast iteration.
    """

    DATASET_ID = "nano-macha"
    CORPUS_ITEM_NAME = "documents"

    def _load_dataset(self) -> None:
        """Load the nano Macha corpus and questions."""
        print("Loading Nano Macha datasets...")
        corpus_data = load_dataset("chonkie-ai/nano-macha", "corpus", split="train")
        questions_data = load_dataset("chonkie-ai/nano-macha", "questions", split="train")

        self.corpus = [doc["text"] for doc in corpus_data]
        self.questions = [q["question"] for q in questions_data]
        self.relevant_passages = [
            q.get("chunk-must-contain", q.get("supporting_passage", "")) for q in questions_data
        ]

        print(f"Loaded {len(self.corpus)} READMEs and {len(self.questions)} questions")


@register_evaluator("nano-cocha")
class NanoCochaEvaluator(BaseEvaluator):
    """Nano evaluator for the Cocha dataset (Code Documents).

    Uses a subset of ~100 questions for fast iteration.
    """

    DATASET_ID = "nano-cocha"
    CORPUS_ITEM_NAME = "files"

    def _load_dataset(self) -> None:
        """Load the nano Cocha corpus and questions."""
        print("Loading Nano Cocha datasets...")
        corpus_data = load_dataset("chonkie-ai/nano-cocha", "corpus", split="train")
        questions_data = load_dataset("chonkie-ai/nano-cocha", "questions", split="train")

        self.corpus = [doc.get("text", doc.get("content", "")) for doc in corpus_data]
        self.questions = [q["question"] for q in questions_data]
        self.relevant_passages = [q.get("chunk-must-contain", "") for q in questions_data]

        print(f"Loaded {len(self.corpus)} code files and {len(self.questions)} questions")

    def _get_extra_metadata(self) -> Dict[str, Any]:
        """Add language metadata for Cocha."""
        return {
            "languages": [
                "python",
                "javascript",
                "typescript",
                "java",
                "go",
                "rust",
                "c++",
                "c",
                "ruby",
                "php",
            ],
        }


@register_evaluator("nano-tacha")
class NanoTachaEvaluator(BaseEvaluator):
    """Nano evaluator for the Tacha dataset (Table-heavy Documents).

    Uses a subset of ~100 questions for fast iteration.
    """

    DATASET_ID = "nano-tacha"
    CORPUS_ITEM_NAME = "documents"

    def _load_dataset(self) -> None:
        """Load the nano Tacha corpus and questions."""
        print("Loading Nano Tacha datasets...")
        corpus_data = load_dataset("chonkie-ai/nano-tacha", "corpus", split="train")
        questions_data = load_dataset("chonkie-ai/nano-tacha", "questions", split="train")

        self.corpus = [doc["text"] for doc in corpus_data]
        self.questions = [q["question"] for q in questions_data]
        self.relevant_passages = [q["chunk-must-contain"] for q in questions_data]

        print(f"Loaded {len(self.corpus)} documents and {len(self.questions)} questions")


@register_evaluator("nano-sencha")
class NanoSenchaEvaluator(BaseEvaluator):
    """Nano evaluator for the Sencha dataset (Scientific Papers).

    Uses a subset of ~100 questions for fast iteration.
    """

    DATASET_ID = "nano-sencha"
    CORPUS_ITEM_NAME = "papers"

    def _load_dataset(self) -> None:
        """Load the nano Sencha corpus and questions."""
        print("Loading Nano Sencha datasets...")
        corpus_data = load_dataset("chonkie-ai/nano-sencha", "corpus", split="train")
        questions_data = load_dataset("chonkie-ai/nano-sencha", "questions", split="train")

        self.corpus = [doc["text"] for doc in corpus_data]
        self.questions = [q["question"] for q in questions_data]
        self.relevant_passages = [q["chunk-must-contain"] for q in questions_data]

        print(f"Loaded {len(self.corpus)} papers and {len(self.questions)} questions")


@register_evaluator("nano-hojicha")
class NanoHojichaEvaluator(BaseEvaluator):
    """Nano evaluator for the Hojicha dataset (Legal Documents).

    Uses a subset of ~100 questions for fast iteration.
    """

    DATASET_ID = "nano-hojicha"
    CORPUS_ITEM_NAME = "contracts"

    def _load_dataset(self) -> None:
        """Load the nano Hojicha corpus and questions."""
        print("Loading Nano Hojicha datasets...")
        corpus_data = load_dataset("chonkie-ai/nano-hojicha", "corpus", split="train")
        questions_data = load_dataset("chonkie-ai/nano-hojicha", "questions", split="train")

        self.corpus = [doc["text"] for doc in corpus_data]
        self.questions = [q["question"] for q in questions_data]
        self.relevant_passages = [q["chunk-must-contain"] for q in questions_data]

        print(f"Loaded {len(self.corpus)} contracts and {len(self.questions)} questions")


@register_evaluator("nano-ryokucha")
class NanoRyokuchaEvaluator(BaseEvaluator):
    """Nano evaluator for the Ryokucha dataset (Conversational/Chat).

    Uses a subset of ~100 questions for fast iteration.
    """

    DATASET_ID = "nano-ryokucha"
    CORPUS_ITEM_NAME = "conversations"

    def _load_dataset(self) -> None:
        """Load the nano Ryokucha corpus and questions."""
        print("Loading Nano Ryokucha datasets...")
        corpus_data = load_dataset("chonkie-ai/nano-ryokucha", "corpus", split="train")
        questions_data = load_dataset("chonkie-ai/nano-ryokucha", "questions", split="train")

        self.corpus = [doc["text"] for doc in corpus_data]
        self.questions = [q["question"] for q in questions_data]
        self.relevant_passages = [q["chunk-must-contain"] for q in questions_data]

        print(f"Loaded {len(self.corpus)} conversations and {len(self.questions)} questions")


@register_evaluator("nano-genmaicha")
class NanoGenmaiichaEvaluator(BaseEvaluator):
    """Nano evaluator for the Genmaicha dataset (Mixed/General).

    Uses a subset of ~100 questions for fast iteration.
    """

    DATASET_ID = "nano-genmaicha"
    CORPUS_ITEM_NAME = "documents"

    def _load_dataset(self) -> None:
        """Load the nano Genmaicha corpus and questions."""
        print("Loading Nano Genmaicha datasets...")
        corpus_data = load_dataset("chonkie-ai/nano-genmaicha", "corpus", split="train")
        questions_data = load_dataset("chonkie-ai/nano-genmaicha", "questions", split="train")

        self.corpus = [doc["text"] for doc in corpus_data]
        self.questions = [q["question"] for q in questions_data]
        self.relevant_passages = [q["chunk-must-contain"] for q in questions_data]

        print(f"Loaded {len(self.corpus)} documents and {len(self.questions)} questions")
