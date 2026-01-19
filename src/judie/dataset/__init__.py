"""Dataset generation module for Judie.

This module provides tools for generating verified QA datasets from documents,
with support for LLM-based semantic deduplication and customizable validation.
"""

from .deduplicator import (
    EmbeddingDeduplicator,
    LLMDeduplicator,
    NoOpDeduplicator,
    SampleDeduplicator,
)
from .generator import DatasetGenerator
from .prompts import DatasetPromptTemplate
from .types import GeneratedSample, GenerationResult
from .validator import (
    CombinedValidator,
    ExactMatchValidator,
    FuzzyMatchValidator,
    SampleValidator,
)

__all__ = [
    # Main generator
    "DatasetGenerator",
    # Types
    "GeneratedSample",
    "GenerationResult",
    # Prompts
    "DatasetPromptTemplate",
    # Validators
    "SampleValidator",
    "ExactMatchValidator",
    "FuzzyMatchValidator",
    "CombinedValidator",
    # Deduplicators
    "SampleDeduplicator",
    "LLMDeduplicator",
    "EmbeddingDeduplicator",
    "NoOpDeduplicator",
]
