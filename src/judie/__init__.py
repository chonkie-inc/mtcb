"""Main module for Judie - Evaluate chunk quality for RAG systems."""

from .cache import EvalCache
from .embeddings import CatsuEmbeddings, get_embeddings, get_tokenizer_for_model, TOKENIZER_MAP
from .evaluators import BaseEvaluator, SimpleEvaluator, GachaEvaluator
from .generator import DatasetGenerator
from .store import SimpleVectorStore
from .types import EvalResult

__version__ = "0.0.3"

__all__ = [
    "__version__",
    # Evaluators
    "BaseEvaluator",
    "SimpleEvaluator",
    "GachaEvaluator",
    # Embeddings
    "CatsuEmbeddings",
    "get_embeddings",
    "get_tokenizer_for_model",
    "TOKENIZER_MAP",
    # Utilities
    "DatasetGenerator",
    "SimpleVectorStore",
    "EvalCache",
    "EvalResult",
]
