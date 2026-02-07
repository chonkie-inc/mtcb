"""Validators for generated samples."""

from typing import Protocol

from .types import GeneratedSample


class SampleValidator(Protocol):
    """Protocol for sample validators."""

    def validate(self, sample: GeneratedSample, chunk_text: str) -> bool:
        """Validate a generated sample against the source chunk.

        Args:
            sample: The generated sample to validate.
            chunk_text: The source chunk text.

        Returns:
            True if the sample is valid, False otherwise.

        """
        ...


class ExactMatchValidator:
    """Validator that requires exact substring match.

    This is the strictest validator - the chunk_must_contain text
    must appear verbatim in the source chunk.
    """

    def validate(self, sample: GeneratedSample, chunk_text: str) -> bool:
        """Check if chunk_must_contain exists exactly in chunk_text."""
        if not sample.chunk_must_contain:
            return False
        return sample.chunk_must_contain in chunk_text


class FuzzyMatchValidator:
    """Validator that allows fuzzy matching with a similarity threshold.

    Uses character-level similarity to handle minor formatting differences.
    """

    def __init__(self, similarity_threshold: float = 0.95):
        """Initialize the fuzzy validator.

        Args:
            similarity_threshold: Minimum similarity ratio (0.0 to 1.0).
                Default 0.95 allows for minor whitespace differences.

        """
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        self.similarity_threshold = similarity_threshold

    def validate(self, sample: GeneratedSample, chunk_text: str) -> bool:
        """Check if chunk_must_contain matches chunk_text with fuzzy matching."""
        if not sample.chunk_must_contain:
            return False

        source = sample.chunk_must_contain

        # First try exact match
        if source in chunk_text:
            return True

        # Try normalized matching (collapse whitespace)
        normalized_source = " ".join(source.split())
        normalized_chunk = " ".join(chunk_text.split())

        if normalized_source in normalized_chunk:
            return True

        # Fall back to sliding window similarity
        return self._sliding_window_match(source, chunk_text)

    def _sliding_window_match(self, source: str, text: str) -> bool:
        """Check if source matches any window of text above threshold."""
        source_len = len(source)
        if source_len == 0 or len(text) < source_len:
            return False

        # Slide a window of source length across the text
        for i in range(len(text) - source_len + 1):
            window = text[i : i + source_len]
            similarity = self._char_similarity(source, window)
            if similarity >= self.similarity_threshold:
                return True

        return False

    def _char_similarity(self, a: str, b: str) -> float:
        """Calculate character-level similarity ratio."""
        if len(a) != len(b):
            return 0.0
        if not a:
            return 1.0

        matches = sum(1 for c1, c2 in zip(a, b) if c1 == c2)
        return matches / len(a)


class CombinedValidator:
    """Validator that tries multiple validators in order.

    Returns True if any validator passes.
    """

    def __init__(self, validators: list[SampleValidator]):
        """Initialize with a list of validators to try."""
        self.validators = validators

    def validate(self, sample: GeneratedSample, chunk_text: str) -> bool:
        """Return True if any validator passes."""
        return any(v.validate(sample, chunk_text) for v in self.validators)
