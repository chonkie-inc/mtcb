"""Main module for Judie."""

from .evaluators import BaseEvaluator, SimpleEvaluator, GachaEvaluator
from .generator import DatasetGenerator
from .store import SimpleVectorStore

__version__ = "0.0.1"
__all__ = ["__version__"]

# Add all the evaluators to the __all__ list
__all__ += [
    "BaseEvaluator",
    "SimpleEvaluator", 
    "GachaEvaluator",
    "DatasetGenerator",
    "SimpleVectorStore",
]