"""Embedding providers for Judie using Catsu."""

from typing import List, Optional, Union

import numpy as np

# Tokenizer mapping: embedding model -> tokenizer identifier
TOKENIZER_MAP = {
    # Voyage AI models (HuggingFace tokenizers)
    "voyage-3-large": "voyageai/voyage-3-large",
    "voyage-3": "voyageai/voyage-3",
    "voyage-3-lite": "voyageai/voyage-3-lite",
    "voyage-code-3": "voyageai/voyage-code-3",
    "voyage-finance-2": "voyageai/voyage-finance-2",
    "voyage-law-2": "voyageai/voyage-law-2",
    "voyage-code-2": "voyageai/voyage-code-2",
    # OpenAI models (tiktoken)
    "text-embedding-3-large": "cl100k_base",
    "text-embedding-3-small": "cl100k_base",
    "text-embedding-ada-002": "cl100k_base",
    # Cohere models
    "embed-english-v3.0": "Cohere/Cohere-embed-english-v3.0",
    "embed-multilingual-v3.0": "Cohere/Cohere-embed-multilingual-v3.0",
    "embed-english-light-v3.0": "Cohere/Cohere-embed-english-light-v3.0",
    "embed-multilingual-light-v3.0": "Cohere/Cohere-embed-multilingual-light-v3.0",
    # Jina models
    "jina-embeddings-v3": "jinaai/jina-embeddings-v3",
    "jina-embeddings-v2-base-en": "jinaai/jina-embeddings-v2-base-en",
}

# Default fallback tokenizer
DEFAULT_TOKENIZER = "gpt2"


def get_tokenizer_for_model(model: str, override: Optional[str] = None):
    """Get the appropriate tokenizer for an embedding model.

    Args:
        model: The embedding model name (e.g., "voyage-3-large")
        override: Optional explicit tokenizer to use instead of auto-detection

    Returns:
        Initialized tokenizer object

    """
    if override and override != "auto":
        tokenizer_id = override
    else:
        # Strip provider prefix if present (e.g., "openai:text-embedding-3-large")
        model_name = model.split(":")[-1] if ":" in model else model
        tokenizer_id = TOKENIZER_MAP.get(model_name, DEFAULT_TOKENIZER)

    # Load the tokenizer based on type
    if tokenizer_id.startswith("cl100k") or tokenizer_id.startswith("p50k") or tokenizer_id.startswith("r50k"):
        # tiktoken tokenizer (OpenAI)
        import tiktoken
        return tiktoken.get_encoding(tokenizer_id)
    else:
        # HuggingFace tokenizer
        from tokenizers import Tokenizer
        return Tokenizer.from_pretrained(tokenizer_id)


def get_tokenizer_id_for_model(model: str, override: Optional[str] = None) -> str:
    """Get the tokenizer identifier string for an embedding model.

    Args:
        model: The embedding model name
        override: Optional explicit tokenizer to use

    Returns:
        Tokenizer identifier string

    """
    if override and override != "auto":
        return override
    model_name = model.split(":")[-1] if ":" in model else model
    return TOKENIZER_MAP.get(model_name, DEFAULT_TOKENIZER)


class CatsuEmbeddings:
    """Embedding provider using Catsu for unified access to multiple providers.

    Catsu supports Voyage AI, OpenAI, Cohere, Jina, and other embedding providers
    through a single unified interface.

    Args:
        model: The embedding model to use (e.g., "voyage-3-large", "text-embedding-3-large")
        batch_size: Maximum batch size for embedding requests (default: auto-calculated)

    Example:
        >>> embeddings = CatsuEmbeddings(model="voyage-3-large")
        >>> vec = embeddings.embed("Hello world")
        >>> vecs = embeddings.embed_batch(["Hello", "World"])

    """

    def __init__(
        self,
        model: str = "voyage-3-large",
        batch_size: Optional[int] = None,
    ) -> None:
        """Initialize CatsuEmbeddings.

        Args:
            model: The embedding model to use
            batch_size: Optional fixed batch size (auto-calculated if None)

        """
        try:
            import catsu
        except ImportError:
            raise ImportError(
                "catsu is required for CatsuEmbeddings. "
                "Install it with: pip install catsu"
            )

        self.model = model
        self.client = catsu.Client()
        self._batch_size = batch_size
        self._dimension: Optional[int] = None

    @property
    def batch_size(self) -> int:
        """Get the batch size for embedding requests."""
        if self._batch_size is not None:
            return self._batch_size
        # Auto-calculate based on model (assuming ~100k token limit)
        # Most models have ~8k token context, so 128 is safe default
        return 128

    @property
    def dimension(self) -> int:
        """Get the embedding dimension (lazy-loaded)."""
        if self._dimension is None:
            # Get dimension by embedding a test string
            test_emb = self.embed("test")
            self._dimension = len(test_emb)
        return self._dimension

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text.

        Args:
            text: The text to embed

        Returns:
            Embedding vector as numpy array

        """
        response = self.client.embed(
            model=self.model,
            input=[text],
            input_type="query",
        )
        return np.array(response.embeddings[0])

    def embed_batch(
        self,
        texts: List[str],
        input_type: str = "document",
        show_progress: bool = False,
    ) -> List[np.ndarray]:
        """Embed a batch of texts.

        Args:
            texts: List of texts to embed
            input_type: Type of input ("document" or "query")
            show_progress: Whether to show progress bar

        Returns:
            List of embedding vectors as numpy arrays

        """
        all_embeddings = []
        batch_size = self.batch_size

        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(
                range(0, len(texts), batch_size),
                desc="Embedding",
                leave=False
            )
        else:
            iterator = range(0, len(texts), batch_size)

        for i in iterator:
            batch = texts[i:i + batch_size]
            response = self.client.embed(
                model=self.model,
                input=batch,
                input_type=input_type,
            )
            all_embeddings.extend([np.array(emb) for emb in response.embeddings])

        return all_embeddings

    def __repr__(self) -> str:
        """String representation."""
        return f"CatsuEmbeddings(model={self.model!r})"

    def __str__(self) -> str:
        """String representation."""
        return f"CatsuEmbeddings({self.model})"


def get_embeddings(
    model: Union[str, "CatsuEmbeddings", any],
    **kwargs,
) -> Union["CatsuEmbeddings", any]:
    """Get an embeddings instance for the given model.

    This is a factory function that returns the appropriate embeddings class.
    If a string is provided, it creates a CatsuEmbeddings instance.
    If an embeddings object is provided, it returns it as-is.

    Args:
        model: Model name string or embeddings instance
        **kwargs: Additional arguments passed to CatsuEmbeddings

    Returns:
        Embeddings instance

    Example:
        >>> emb = get_embeddings("voyage-3-large")
        >>> emb = get_embeddings(my_custom_embeddings)  # pass-through

    """
    if isinstance(model, str):
        # Check for legacy model2vec:// prefix for backward compatibility
        if model.startswith("model2vec://"):
            from chonkie import AutoEmbeddings
            return AutoEmbeddings.get_embeddings(model)
        # Use Catsu for all other models
        return CatsuEmbeddings(model=model, **kwargs)
    else:
        # Assume it's already an embeddings instance
        return model
