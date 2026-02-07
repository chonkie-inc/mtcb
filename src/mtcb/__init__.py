"""Main module for MTCB - Massive Text Chunking Benchmark. Evaluate chunk quality for RAG systems."""

from .benchmark import (
    AVAILABLE_METRICS,
    Benchmark,
    BenchmarkResult,
    NanoBenchmark,
    get_available_datasets,
    get_full_datasets,
    get_nano_datasets,
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
from .embeddings import TOKENIZER_MAP, CatsuEmbeddings, get_embeddings, get_tokenizer_for_model
from .evaluators import (
    BaseEvaluator,
    CochaEvaluator,
    FichaEvaluator,
    # Full evaluators
    GachaEvaluator,
    GenmaiichaEvaluator,
    HojichaEvaluator,
    MachaEvaluator,
    NanoCochaEvaluator,
    NanoFichaEvaluator,
    # Nano evaluators
    NanoGachaEvaluator,
    NanoGenmaiichaEvaluator,
    NanoHojichaEvaluator,
    NanoMachaEvaluator,
    NanoRyokuchaEvaluator,
    NanoSenchaEvaluator,
    NanoTachaEvaluator,
    RyokuchaEvaluator,
    SenchaEvaluator,
    SimpleEvaluator,
    TachaEvaluator,
)
from .store import SimpleVectorStore
from .types import EvalResult

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # Benchmark
    "AVAILABLE_METRICS",
    "Benchmark",
    "BenchmarkResult",
    "NanoBenchmark",
    "get_available_datasets",
    "get_nano_datasets",
    "get_full_datasets",
    # Full Evaluators
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
    # Nano Evaluators
    "NanoGachaEvaluator",
    "NanoFichaEvaluator",
    "NanoMachaEvaluator",
    "NanoCochaEvaluator",
    "NanoTachaEvaluator",
    "NanoSenchaEvaluator",
    "NanoHojichaEvaluator",
    "NanoRyokuchaEvaluator",
    "NanoGenmaiichaEvaluator",
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
