"""Types for Judie."""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Union


@dataclass
class EvalResult:
    """Results from an evaluation run with metrics, metadata, and performance statistics.
    
    A focused container for evaluation results that provides structured access
    to metrics and utility methods for saving/loading.
    """
    
    # Core metrics - extensible dictionary
    metrics: Dict[str, Dict[int, float]]  # e.g., {"recall": {1: 0.33, 3: 0.49}}
    
    # Metadata about the evaluation
    metadata: Dict[str, Any]
    
    # Performance statistics
    total_corpus_size_mb: float
    total_time_to_chunk: float
    chunk_speed_mb_per_sec: float
    total_chunks_created: int
    total_evaluation_time: float
    questions_per_second: float
    
    # Auto-generated timestamp
    timestamp: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Validate the EvalResult after initialization."""
        if "recall" not in self.metrics and "mrr" not in self.metrics:
            raise ValueError("EvalResult must contain at least 'recall' or 'mrr' metrics")

    @property
    def recall(self) -> Dict[int, float]:
        """Get recall metrics for easy access."""
        return self.metrics.get("recall", {})

    @property
    def mrr(self) -> Dict[int, float]:
        """Get MRR metrics for easy access."""
        return self.metrics.get("mrr", {})

    @property
    def k_values(self) -> List[int]:
        """Get all k values that were evaluated."""
        if "recall" in self.metrics:
            return sorted(self.metrics["recall"].keys())
        elif "mrr" in self.metrics:
            return sorted(self.metrics["mrr"].keys())
        return []
    
    def get_metric(self, metric_name: str, k: int) -> float:
        """Get a specific metric value for a given k."""
        return self.metrics[metric_name][k]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return {
            "metrics": self.metrics,
            "metadata": self.metadata,
            "total_corpus_size_mb": self.total_corpus_size_mb,
            "total_time_to_chunk": self.total_time_to_chunk,
            "chunk_speed_mb_per_sec": self.chunk_speed_mb_per_sec,
            "total_chunks_created": self.total_chunks_created,
            "total_evaluation_time": self.total_evaluation_time,
            "questions_per_second": self.questions_per_second,
            "timestamp": self.timestamp,
        }
    
    def save(self, path: Union[str, Path]) -> None:
        """Save results to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "EvalResult":
        """Load results from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    def __str__(self) -> str:
        """Pretty string representation of results."""
        lines = []
        lines.append("="*60)
        lines.append("EVALUATION RESULTS")
        lines.append("="*60)
        
        # Chunker info from metadata
        if "chunker_type" in self.metadata:
            lines.append(f"Chunker: {self.metadata['chunker_type']}")
        if "chunker_config" in self.metadata:
            lines.append(f"Config: {self.metadata['chunker_config']}")
        lines.append("-" * 60)
        
        # Performance statistics
        lines.append(f"Corpus size: {self.total_corpus_size_mb:.2f} MB")
        lines.append(f"Chunks created: {self.total_chunks_created:,}")
        lines.append(f"Chunking time: {self.total_time_to_chunk:.2f}s")
        lines.append(f"Chunking speed: {self.chunk_speed_mb_per_sec:.2f} MB/s")
        lines.append(f"Evaluation time: {self.total_evaluation_time:.2f}s")
        lines.append(f"Questions/sec: {self.questions_per_second:.1f}")
        lines.append("-" * 60)
        
        # Metrics
        if "recall" in self.metrics:
            lines.append("Recall@k:")
            for k in self.k_values:
                recall = self.metrics["recall"][k]
                lines.append(f"  k={k:2d}: {recall:.2%}")

        if "mrr" in self.metrics:
            lines.append("MRR@k:")
            for k in sorted(self.metrics["mrr"].keys()):
                mrr_val = self.metrics["mrr"][k]
                lines.append(f"  k={k:2d}: {mrr_val:.4f}")

        lines.append("="*60)
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        """Concise representation."""
        parts = []
        if "recall" in self.metrics:
            recall_summary = ", ".join([f"R@{k}={v:.2%}" for k, v in sorted(self.metrics["recall"].items())])
            parts.append(recall_summary)
        if "mrr" in self.metrics:
            # Just show the max k MRR value for brevity
            max_k = max(self.metrics["mrr"].keys())
            parts.append(f"MRR@{max_k}={self.metrics['mrr'][max_k]:.4f}")
        chunker_type = self.metadata.get("chunker_type", "Unknown")
        return f"EvalResult({', '.join(parts)}, {chunker_type})"

