"""Main module for Judie - Evaluate chunk quality for RAG systems."""

from .cache import EvalCache
from .dataset import (
    DatasetGenerator,
    DatasetPromptTemplate,
    ExactMatchValidator,
    GeneratedSample,
    GenerationResult,
    LLMDeduplicator,
)
from .embeddings import CatsuEmbeddings, get_embeddings, get_tokenizer_for_model, TOKENIZER_MAP
from .evaluators import BaseEvaluator, SimpleEvaluator, GachaEvaluator
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
    # Dataset generation
    "DatasetGenerator",
    "DatasetPromptTemplate",
    "GeneratedSample",
    "GenerationResult",
    "ExactMatchValidator",
    "LLMDeduplicator",
    # Utilities
    "SimpleVectorStore",
    "EvalCache",
    "EvalResult",
]
