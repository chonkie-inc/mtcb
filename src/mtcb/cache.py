"""Caching utilities for MTCB evaluations."""

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar

import numpy as np

T = TypeVar("T")


def compute_hash(data: str, max_length: int = 16) -> str:
    """Compute a short hash of the given data.

    Args:
        data: String data to hash
        max_length: Maximum length of the returned hash

    Returns:
        Hex string hash

    """
    return hashlib.sha256(data.encode()).hexdigest()[:max_length]


class EvalCache:
    """Cache for evaluation artifacts (chunks, embeddings).

    Provides disk-based caching to avoid re-computation when running
    multiple evaluations with the same chunker/embedding configurations.

    Args:
        cache_dir: Directory to store cache files

    Example:
        >>> cache = EvalCache("./cache")
        >>> chunks = cache.get_or_compute_chunks(
        ...     corpus_id="gacha",
        ...     chunker_config="RecursiveChunker(1000)",
        ...     compute_fn=lambda: chunker.chunk(text)
        ... )

    """

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        """Initialize the cache.

        Args:
            cache_dir: Directory for cache files. If None, caching is disabled.

        """
        self.enabled = cache_dir is not None
        if self.enabled:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None

    def _get_cache_key(self, *parts: str) -> str:
        """Generate a cache key from parts."""
        combined = "_".join(str(p) for p in parts)
        return compute_hash(combined)

    def _chunks_path(self, cache_key: str) -> Path:
        """Get path for chunks cache file."""
        return self.cache_dir / f"chunks_{cache_key}.jsonl"

    def _embeddings_path(self, cache_key: str) -> Path:
        """Get path for embeddings cache file."""
        return self.cache_dir / f"embeddings_{cache_key}.npy"

    def get_or_compute_chunks(
        self,
        corpus_id: str,
        chunker_config: str,
        compute_fn: Callable[[], List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Get chunks from cache or compute them.

        Args:
            corpus_id: Identifier for the corpus (e.g., "gacha", hash of custom corpus)
            chunker_config: String representation of chunker config
            compute_fn: Function to compute chunks if not cached

        Returns:
            List of chunk dictionaries with 'text' and metadata

        """
        if not self.enabled:
            return compute_fn()

        cache_key = self._get_cache_key(corpus_id, chunker_config)
        cache_path = self._chunks_path(cache_key)

        if cache_path.exists():
            # Load from cache
            chunks = []
            with open(cache_path, "r") as f:
                for line in f:
                    chunks.append(json.loads(line))
            return chunks

        # Compute and cache
        chunks = compute_fn()
        with open(cache_path, "w") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk) + "\n")

        return chunks

    def get_or_compute_embeddings(
        self,
        chunks_key: str,
        model: str,
        compute_fn: Callable[[], List[np.ndarray]],
    ) -> List[np.ndarray]:
        """Get embeddings from cache or compute them.

        Args:
            chunks_key: Key identifying the chunks (e.g., corpus_id + chunker_config)
            model: Embedding model name
            compute_fn: Function to compute embeddings if not cached

        Returns:
            List of embedding vectors as numpy arrays

        """
        if not self.enabled:
            return compute_fn()

        cache_key = self._get_cache_key(chunks_key, model)
        cache_path = self._embeddings_path(cache_key)

        if cache_path.exists():
            # Load from cache
            embeddings_array = np.load(cache_path)
            return [embeddings_array[i] for i in range(len(embeddings_array))]

        # Compute and cache
        embeddings = compute_fn()
        np.save(cache_path, np.array(embeddings))

        return embeddings

    def get_or_compute_question_embeddings(
        self,
        dataset_id: str,
        model: str,
        compute_fn: Callable[[], List[np.ndarray]],
    ) -> List[np.ndarray]:
        """Get question embeddings from cache or compute them.

        Args:
            dataset_id: Identifier for the question dataset
            model: Embedding model name
            compute_fn: Function to compute embeddings if not cached

        Returns:
            List of embedding vectors as numpy arrays

        """
        if not self.enabled:
            return compute_fn()

        cache_key = self._get_cache_key(f"questions_{dataset_id}", model)
        cache_path = self._embeddings_path(cache_key)

        if cache_path.exists():
            embeddings_array = np.load(cache_path)
            return [embeddings_array[i] for i in range(len(embeddings_array))]

        embeddings = compute_fn()
        np.save(cache_path, np.array(embeddings))

        return embeddings

    def clear(self) -> None:
        """Clear all cached files."""
        if not self.enabled:
            return

        for path in self.cache_dir.glob("*.jsonl"):
            path.unlink()
        for path in self.cache_dir.glob("*.npy"):
            path.unlink()

    def __repr__(self) -> str:
        """String representation."""
        if self.enabled:
            return f"EvalCache(cache_dir={self.cache_dir!r})"
        return "EvalCache(disabled)"
