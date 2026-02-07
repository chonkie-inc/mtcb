"""Types for dataset generation."""

from dataclasses import dataclass, field
from typing import List, Optional

from pydantic import BaseModel


class GeneratedSample(BaseModel):
    """A generated QA sample with source verification.

    Attributes:
        question: The generated question.
        answer: The answer to the question.
        chunk_must_contain: Exact text span that must exist in the source chunk.
        document_id: ID of the source document.
        chunk_id: ID of the source chunk.
        verified: Whether the chunk_must_contain was validated.

    """

    question: str
    answer: str
    chunk_must_contain: str
    document_id: Optional[int] = None
    chunk_id: Optional[str] = None
    verified: bool = False


@dataclass
class GenerationResult:
    """Result of dataset generation with statistics.

    Attributes:
        samples: List of generated and verified samples.
        total_generated: Total number of samples generated (before validation).
        total_verified: Number of samples that passed validation.
        failed_validation_count: Number of samples that failed validation.
        duplicate_count: Number of samples removed as duplicates.
        generation_time_seconds: Total time taken for generation.

    """

    samples: List[GeneratedSample] = field(default_factory=list)
    total_generated: int = 0
    total_verified: int = 0
    failed_validation_count: int = 0
    duplicate_count: int = 0
    generation_time_seconds: float = 0.0

    def __repr__(self) -> str:
        return (
            f"GenerationResult("
            f"samples={len(self.samples)}, "
            f"verified={self.total_verified}, "
            f"failed={self.failed_validation_count}, "
            f"duplicates={self.duplicate_count}, "
            f"time={self.generation_time_seconds:.2f}s)"
        )
