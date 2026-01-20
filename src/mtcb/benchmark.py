"""Benchmark module for running evaluations across multiple datasets."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from .types import EvalResult


# Registry of available datasets and their evaluator classes
_EVALUATOR_REGISTRY: Dict[str, type] = {}


def register_evaluator(name: str):
    """Decorator to register an evaluator class for a dataset.

    Args:
        name: The dataset name (e.g., "gacha", "macha", "ficha")

    Example:
        @register_evaluator("gacha")
        class GachaEvaluator(BaseEvaluator):
            ...
    """
    def decorator(cls):
        _EVALUATOR_REGISTRY[name.lower()] = cls
        return cls
    return decorator


def get_available_datasets() -> List[str]:
    """Get list of available dataset names."""
    return list(_EVALUATOR_REGISTRY.keys())


def get_evaluator_class(dataset: str) -> type:
    """Get the evaluator class for a dataset.

    Args:
        dataset: Dataset name (case-insensitive)

    Returns:
        The evaluator class for the dataset

    Raises:
        ValueError: If dataset is not registered
    """
    dataset_lower = dataset.lower()
    if dataset_lower not in _EVALUATOR_REGISTRY:
        available = ", ".join(get_available_datasets())
        raise ValueError(
            f"Unknown dataset: '{dataset}'. Available datasets: {available}"
        )
    return _EVALUATOR_REGISTRY[dataset_lower]


@dataclass
class BenchmarkResult:
    """Results from running a benchmark across multiple datasets.

    Attributes:
        name: Name of the benchmark
        results: Dictionary mapping dataset names to their EvalResult
        chunker_config: String representation of the chunker used
        embedding_model: Name of the embedding model used
        k_values: The k values that were evaluated
    """

    name: str
    results: Dict[str, EvalResult] = field(default_factory=dict)
    chunker_config: str = ""
    embedding_model: str = ""
    k_values: List[int] = field(default_factory=list)

    @property
    def datasets(self) -> List[str]:
        """Get list of datasets in this result."""
        return list(self.results.keys())

    @property
    def mean_recall(self) -> Dict[int, float]:
        """Get mean recall@k across all datasets."""
        if not self.results:
            return {}

        mean_recall = {}
        for k in self.k_values:
            recalls = [
                r.metrics.get("recall", {}).get(k, 0.0)
                for r in self.results.values()
            ]
            mean_recall[k] = sum(recalls) / len(recalls) if recalls else 0.0
        return mean_recall

    @property
    def mean_mrr(self) -> Dict[int, float]:
        """Get mean MRR@k across all datasets."""
        if not self.results:
            return {}

        mean_mrr = {}
        for k in self.k_values:
            mrrs = [
                r.metrics.get("mrr", {}).get(k, 0.0)
                for r in self.results.values()
            ]
            mean_mrr[k] = sum(mrrs) / len(mrrs) if mrrs else 0.0
        return mean_mrr

    @property
    def total_evaluation_time(self) -> float:
        """Get total evaluation time across all datasets."""
        return sum(r.total_evaluation_time for r in self.results.values())

    @property
    def total_chunks_created(self) -> int:
        """Get total chunks created across all datasets."""
        return sum(r.total_chunks_created for r in self.results.values())

    @property
    def total_corpus_size_mb(self) -> float:
        """Get total corpus size across all datasets."""
        return sum(r.total_corpus_size_mb for r in self.results.values())

    def get_result(self, dataset: str) -> Optional[EvalResult]:
        """Get the result for a specific dataset."""
        return self.results.get(dataset.lower())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "chunker_config": self.chunker_config,
            "embedding_model": self.embedding_model,
            "k_values": self.k_values,
            "mean_recall": self.mean_recall,
            "mean_mrr": self.mean_mrr,
            "total_evaluation_time": self.total_evaluation_time,
            "total_chunks_created": self.total_chunks_created,
            "total_corpus_size_mb": self.total_corpus_size_mb,
            "results": {
                name: result.to_dict()
                for name, result in self.results.items()
            },
        }

    def __str__(self) -> str:
        """Pretty string representation."""
        lines = []
        lines.append("=" * 70)
        lines.append(f"{self.name} BENCHMARK RESULTS")
        lines.append("=" * 70)
        lines.append(f"Chunker: {self.chunker_config}")
        lines.append(f"Embedding: {self.embedding_model}")
        lines.append("-" * 70)

        # Header row
        k_cols = "".join(f"{'R@' + str(k):>10}" for k in self.k_values)
        mrr_cols = "".join(f"{'MRR@' + str(k):>10}" for k in self.k_values)
        lines.append(f"{'Dataset':<15}{k_cols}{mrr_cols}")
        lines.append("-" * 70)

        # Per-dataset rows
        for dataset, result in self.results.items():
            recall_vals = "".join(
                f"{result.metrics.get('recall', {}).get(k, 0.0):>10.2%}"
                for k in self.k_values
            )
            mrr_vals = "".join(
                f"{result.metrics.get('mrr', {}).get(k, 0.0):>10.4f}"
                for k in self.k_values
            )
            lines.append(f"{dataset.capitalize():<15}{recall_vals}{mrr_vals}")

        # Mean row
        lines.append("-" * 70)
        mean_recall_vals = "".join(
            f"{self.mean_recall.get(k, 0.0):>10.2%}"
            for k in self.k_values
        )
        mean_mrr_vals = "".join(
            f"{self.mean_mrr.get(k, 0.0):>10.4f}"
            for k in self.k_values
        )
        lines.append(f"{'MEAN':<15}{mean_recall_vals}{mean_mrr_vals}")

        # Stats
        lines.append("-" * 70)
        lines.append(f"Total time: {self.total_evaluation_time:.2f}s")
        lines.append(f"Total chunks: {self.total_chunks_created:,}")
        lines.append(f"Corpus size: {self.total_corpus_size_mb:.2f} MB")
        lines.append("=" * 70)

        return "\n".join(lines)

    def __repr__(self) -> str:
        """Concise representation."""
        datasets_str = ", ".join(self.datasets)
        mean_r5 = self.mean_recall.get(5, self.mean_recall.get(1, 0.0))
        return f"BenchmarkResult({self.name}, datasets=[{datasets_str}], mean_R@5={mean_r5:.2%})"


@dataclass
class Benchmark:
    """A benchmark that runs evaluation across multiple datasets.

    A Benchmark groups multiple datasets (like Gacha, Macha, Ficha) together
    and provides a single interface to evaluate a chunker across all of them.

    By default, creates the full MTCB benchmark with all available datasets.

    Args:
        name: Name of the benchmark (default: "Massive Text Chunking Benchmark")
        datasets: List of dataset names to include. If None, uses all available datasets.
        description: Optional description of the benchmark
        reference: Optional URL reference (paper, leaderboard, etc.)

    Example:
        >>> from mtcb import Benchmark
        >>> from chonkie import RecursiveChunker
        >>>
        >>> # Default: full MTCB benchmark with all datasets
        >>> benchmark = Benchmark()
        >>>
        >>> # Or specify specific datasets
        >>> benchmark = Benchmark(datasets=["gacha", "ficha"])
        >>>
        >>> # Run the benchmark
        >>> result = benchmark.evaluate(
        ...     chunker=RecursiveChunker(chunk_size=1000),
        ...     embedding_model="voyage-3-large",
        ...     k=[1, 3, 5, 10],
        ... )
        >>> print(result)
    """

    name: str = "Massive Text Chunking Benchmark"
    datasets: Optional[List[str]] = None
    description: Optional[str] = None
    reference: Optional[str] = None

    def __post_init__(self):
        """Set defaults and validate datasets exist."""
        # Default to all available datasets
        if self.datasets is None:
            self.datasets = get_available_datasets()

        # Validate all datasets exist
        for dataset in self.datasets:
            if dataset.lower() not in _EVALUATOR_REGISTRY:
                available = ", ".join(get_available_datasets())
                raise ValueError(
                    f"Unknown dataset: '{dataset}'. Available: {available}"
                )

    def evaluate(
        self,
        chunker: Any,
        embedding_model: Union[str, Any] = "voyage-3-large",
        k: Union[int, List[int]] = [1, 3, 5, 10],
        tokenizer: Optional[str] = "auto",
        cache_dir: Optional[str] = None,
        show_progress_bar: bool = True,
    ) -> BenchmarkResult:
        """Run the benchmark across all datasets.

        Args:
            chunker: The chunker to evaluate
            embedding_model: Embedding model to use (default: "voyage-3-large")
            k: K values for recall/MRR evaluation
            tokenizer: Tokenizer for chunking ("auto" to detect from model)
            cache_dir: Directory for caching (None to disable)
            show_progress_bar: Whether to show progress bars

        Returns:
            BenchmarkResult containing results for all datasets
        """
        k_values = [k] if isinstance(k, int) else k

        results = {}

        print(f"\n{'=' * 60}")
        print(f"Running {self.name} Benchmark")
        print(f"Datasets: {', '.join(self.datasets)}")
        print(f"{'=' * 60}\n")

        for dataset in self.datasets:
            print(f"\n--- Evaluating {dataset.upper()} ---\n")

            # Get the evaluator class and instantiate it
            evaluator_cls = get_evaluator_class(dataset)
            evaluator = evaluator_cls(
                chunker=chunker,
                embedding_model=embedding_model,
                tokenizer=tokenizer,
                cache_dir=cache_dir,
            )

            # Run evaluation
            result = evaluator.evaluate(k=k_values, show_progress_bar=show_progress_bar)
            results[dataset.lower()] = result

        # Build benchmark result
        chunker_config = repr(chunker)
        model_name = embedding_model if isinstance(embedding_model, str) else str(embedding_model)

        benchmark_result = BenchmarkResult(
            name=self.name,
            results=results,
            chunker_config=chunker_config,
            embedding_model=model_name,
            k_values=k_values,
        )

        print(f"\n{benchmark_result}")

        return benchmark_result

    def __len__(self) -> int:
        """Number of datasets in the benchmark."""
        return len(self.datasets)

    def __iter__(self):
        """Iterate over dataset names."""
        return iter(self.datasets)

    def __repr__(self) -> str:
        return f"Benchmark({self.name!r}, datasets={self.datasets})"


