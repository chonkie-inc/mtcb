"""This module contains the SimpleVectorStore class for storing and retrieving embeddings."""

from typing import Any, List, Optional, Tuple

import numpy as np


class SimpleVectorStore:
    """A simple in-memory vector store that stores embeddings and metadata.

    Supports searching for the top K nearest neighbors using cosine similarity.
    """

    def __init__(self):
        self.embeddings = None  # Will store embeddings as a numpy array
        self.metadata = []  # Will store metadata as a list of dicts
        self._normalized_embeddings = None  # To store normalized embeddings for efficient search

    def add(self, embedding: np.ndarray, metadata: Optional[Any] = None) -> None:
        """Adds an embedding and its associated metadata to the store.

        Args:
            embedding (np.ndarray): The embedding vector.
            metadata (dict, optional): The metadata for the embedding. Defaults to None.

        """
        if self.embeddings is None:
            self.embeddings = embedding.reshape(1, -1)
        else:
            self.embeddings = np.vstack((self.embeddings, embedding))

        self.metadata.append(metadata)
        self._normalized_embeddings = None  # Invalidate normalized embeddings

    def add_batch(
        self, embeddings: List[np.ndarray], metadata: Optional[List[Any]] = None
    ) -> None:
        """Adds a batch of embeddings and their associated metadata to the store.

        Args:
            embeddings (List[np.ndarray]): The list of embedding vectors.
            metadata (List[dict], optional): The list of metadata for each embedding. Defaults to None.

        """
        if self.embeddings is None:
            self.embeddings = np.array(embeddings)
        else:
            self.embeddings = np.vstack((self.embeddings, embeddings))

        if metadata is not None:
            self.metadata.extend(metadata)

        self._normalized_embeddings = None  # Invalidate normalized embeddings

    def _normalize_embeddings(self) -> None:
        """Normalizes the stored embeddings."""
        if self.embeddings is not None:
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            # Handle case where norm is zero (e.g., zero vector)
            norms[norms == 0] = 1e-10  # Add a small epsilon to avoid division by zero
            self._normalized_embeddings = self.embeddings / norms
        else:
            self._normalized_embeddings = None

    def query(
        self, query_embedding: np.ndarray, k: int = 5, return_score: bool = False
    ) -> List[Tuple[int, float, Any]]:
        """Searches for the top K nearest neighbors to the query embedding.

        Args:
            query_embedding (np.ndarray): The query embedding vector.
            k (int, optional): The number of neighbors to return. Defaults to 5.

        Returns:
            List[tuple[int, float, dict]]: A list of tuples, where each tuple contains
                                           (index, similarity_score, metadata).
                                           The results are sorted by similarity score in descending order.

        """
        if self.embeddings is None or len(self.embeddings) == 0:
            return []

        if self._normalized_embeddings is None:
            self._normalize_embeddings()

        # Normalize the query embedding
        query_norm = np.linalg.norm(query_embedding)
        # Handle case where query norm is zero
        if query_norm == 0:
            return []
        normalized_query = query_embedding / query_norm

        # Compute cosine similarity: (normalized_query @ normalized_embeddings.T)
        # This is equivalent to (query_embedding @ embeddings.T) / (norm(query) * norm(embeddings))
        assert self._normalized_embeddings is not None
        similarity_scores = normalized_query @ self._normalized_embeddings.T

        # Get the indices of the top K scores
        top_k_indices = np.argsort(similarity_scores)[::-1][:k]

        # Get the results with scores and metadata
        results = []
        for idx in top_k_indices:
            if return_score:
                results.append((idx.item(), similarity_scores[idx], self.metadata[idx]))
            else:
                results.append((idx.item(), None, self.metadata[idx]))

        return results

    def __repr__(self) -> str:
        return f"SimpleVectorStore(embeddings={self.embeddings}, metadata={self.metadata})"
