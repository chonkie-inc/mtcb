"""Base class for all evaluators."""

from abc import ABC, abstractmethod


class BaseEvaluator(ABC):
    """Base class for all evaluators."""

    @abstractmethod
    def evaluate(self, chunk: str) -> float:
        """Evaluate the chunk."""
        pass