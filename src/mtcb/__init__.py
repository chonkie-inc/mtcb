"""Main module for MTCB - Make That Chunker Better! Evaluate chunk quality for RAG systems."""

from .benchmark import (
    AVAILABLE_METRICS,
    Benchmark,
    BenchmarkResult,
    get_available_datasets,
)
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
from .evaluators import BaseEvaluator, SimpleEvaluator, GachaEvaluator, MachaEvaluator, FichaEvaluator, CochaEvaluator, TachaEvaluator, SenchaEvaluator, HojichaEvaluator, RyokuchaEvaluator, GenmaiichaEvaluator
from .store import SimpleVectorStore
from .types import EvalResult

__version__ = "0.0.3"

__all__ = [
    "__version__",
    # Benchmark
    "AVAILABLE_METRICS",
    "Benchmark",
    "BenchmarkResult",
    "get_available_datasets",
    # Evaluators
    "BaseEvaluator",
    "SimpleEvaluator",
    "GachaEvaluator",
    "MachaEvaluator",
    "FichaEvaluator",
    "CochaEvaluator",
    "TachaEvaluator",
    "SenchaEvaluator",
    "HojichaEvaluator",
    "RyokuchaEvaluator",
    "GenmaiichaEvaluator",
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
