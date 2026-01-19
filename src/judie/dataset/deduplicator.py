"""Deduplicators for removing semantically similar samples."""

import json
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel

from .types import GeneratedSample

if TYPE_CHECKING:
    from chonkie import BaseGenie


class SampleDeduplicator(Protocol):
    """Protocol for sample deduplicators."""

    def deduplicate(self, samples: list[GeneratedSample]) -> list[GeneratedSample]:
        """Remove duplicate samples.

        Args:
            samples: List of samples to deduplicate.

        Returns:
            List of unique samples.
        """
        ...


class DeduplicationResult(BaseModel):
    """Result from LLM deduplication."""

    groups: list[list[int]]
    unique_indices: list[int]


class LLMDeduplicator:
    """Deduplicator that uses an LLM to identify semantically similar questions.

    Processes samples in batches and can perform multiple passes to catch
    duplicates that span batch boundaries.
    """

    def __init__(
        self,
        genie: "BaseGenie",
        batch_size: int = 50,
        max_passes: int = 3,
        deduplication_prompt: str | None = None,
    ):
        """Initialize the LLM deduplicator.

        Args:
            genie: The LLM genie to use for deduplication.
            batch_size: Number of samples to process in each batch.
            max_passes: Maximum number of deduplication passes.
            deduplication_prompt: Custom prompt template (optional).
        """
        self.genie = genie
        self.batch_size = batch_size
        self.max_passes = max_passes
        self.deduplication_prompt = deduplication_prompt or self._default_prompt()

    def _default_prompt(self) -> str:
        return """Analyze these questions for semantic similarity.
Group questions that ask essentially the same thing, even if worded differently.

Questions:
{questions_json}

Return JSON with:
- "groups": list of groups, each containing indices of similar questions
- "unique_indices": list of one representative index from each group (prefer longer, more specific questions)"""

    def deduplicate(self, samples: list[GeneratedSample]) -> list[GeneratedSample]:
        """Remove semantically duplicate samples using LLM analysis."""
        if len(samples) <= 1:
            return samples

        current_samples = list(samples)

        for pass_num in range(self.max_passes):
            prev_count = len(current_samples)
            current_samples = self._single_pass(current_samples)

            # Stop if no duplicates found in this pass
            if len(current_samples) == prev_count:
                break

        return current_samples

    def _single_pass(self, samples: list[GeneratedSample]) -> list[GeneratedSample]:
        """Perform a single deduplication pass over all samples."""
        if len(samples) <= self.batch_size:
            return self._deduplicate_batch(samples)

        # Process in batches
        unique_samples = []
        for i in range(0, len(samples), self.batch_size):
            batch = samples[i : i + self.batch_size]

            # Include some samples from previous batches for cross-batch dedup
            if unique_samples:
                overlap = unique_samples[-min(10, len(unique_samples)) :]
                combined = overlap + batch
                deduped = self._deduplicate_batch(combined)
                # Only keep new unique samples (not the overlap)
                new_uniques = [s for s in deduped if s not in overlap]
                unique_samples.extend(new_uniques)
            else:
                unique_samples.extend(self._deduplicate_batch(batch))

        return unique_samples

    def _deduplicate_batch(
        self, samples: list[GeneratedSample]
    ) -> list[GeneratedSample]:
        """Deduplicate a single batch of samples."""
        if len(samples) <= 1:
            return samples

        questions_data = [
            {"index": i, "question": s.question} for i, s in enumerate(samples)
        ]
        questions_json = json.dumps(questions_data, indent=2)

        prompt = self.deduplication_prompt.format(questions_json=questions_json)

        try:
            result = self.genie.generate_json(prompt, DeduplicationResult)

            # Extract unique indices
            if "unique_indices" in result:
                unique_indices = set(result["unique_indices"])
                return [
                    samples[i] for i in sorted(unique_indices) if i < len(samples)
                ]

            # Fallback: extract one from each group
            if "groups" in result:
                unique_indices = set()
                for group in result["groups"]:
                    if group:
                        unique_indices.add(group[0])
                return [
                    samples[i] for i in sorted(unique_indices) if i < len(samples)
                ]

        except Exception:
            # On failure, return original samples
            pass

        return samples


class EmbeddingDeduplicator:
    """Deduplicator that uses embeddings to identify similar questions.

    Faster than LLM-based deduplication but may be less accurate for
    semantic similarity.
    """

    def __init__(
        self,
        embedding_model: str = "catsu",
        similarity_threshold: float = 0.92,
    ):
        """Initialize the embedding deduplicator.

        Args:
            embedding_model: Name of the embedding model to use.
            similarity_threshold: Cosine similarity threshold for duplicates.
        """
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self._embeddings = None

    def _get_embeddings(self):
        """Lazy load embeddings model."""
        if self._embeddings is None:
            from ..embeddings import get_embeddings

            self._embeddings = get_embeddings(self.embedding_model)
        return self._embeddings

    def deduplicate(self, samples: list[GeneratedSample]) -> list[GeneratedSample]:
        """Remove duplicate samples based on embedding similarity."""
        if len(samples) <= 1:
            return samples

        embeddings = self._get_embeddings()
        questions = [s.question for s in samples]

        # Get embeddings for all questions
        question_embeddings = embeddings.embed(questions)

        # Find unique samples
        unique_indices = []
        for i, emb_i in enumerate(question_embeddings):
            is_duplicate = False
            for j in unique_indices:
                similarity = self._cosine_similarity(emb_i, question_embeddings[j])
                if similarity >= self.similarity_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_indices.append(i)

        return [samples[i] for i in unique_indices]

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)


class NoOpDeduplicator:
    """Deduplicator that does nothing - returns samples unchanged."""

    def deduplicate(self, samples: list[GeneratedSample]) -> list[GeneratedSample]:
        """Return samples unchanged."""
        return samples
