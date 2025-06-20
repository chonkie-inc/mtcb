"""Base class for all evaluators."""

from abc import ABC, abstractmethod
from typing import List, Union
from ..types import EvalResult


class BaseEvaluator(ABC):
    """Base class for all evaluators."""

    @abstractmethod
    def evaluate(self, k: Union[int, List[int]] = 1, show_progress_bar: bool = True) -> EvalResult:
        """Evaluate the corpus."""
        pass