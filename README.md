<div align="center">

![MTCB Logo](https://github.com/chonkie-inc/mtcb/blob/main/assets/mtcb.png?raw=true)

# 🔬 mtcb ✨

_The benchmark for evaluating chunking strategies in RAG pipelines._

[![PyPI version](https://img.shields.io/pypi/v/mtcb.svg)](https://pypi.org/project/mtcb/)
[![License](https://img.shields.io/github/license/chonkie-inc/mtcb.svg)](https://github.com/chonkie-inc/mtcb/blob/main/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/chonkie-inc/mtcb.svg)](https://github.com/chonkie-inc/mtcb/stargazers)
[![Downloads](https://static.pepy.tech/badge/mtcb)](https://pepy.tech/project/mtcb)

[Installation](#-installation) •
[Quick Start](#-quick-start) •
[Benchmarks](#-available-benchmarks) •
[Usage](#-usage) •
[Metrics](#-metrics)

</div>

MTCB (Massive Text Chunking Benchmark) is a standardized evaluation framework for text chunking in RAG systems. It measures how well your chunking and embedding strategy retrieves relevant passages across **9 diverse domains**, from legal contracts to scientific papers. Built on top of [Chonkie](https://github.com/chonkie-inc/chonkie).

## 📦 Installation

```bash
pip install mtcb
```

## 🚀 Quick Start

Run the lightweight nano benchmark to evaluate a chunking strategy in minutes:

```python
from mtcb import NanoBenchmark
from chonkie import RecursiveChunker

benchmark = NanoBenchmark()
result = benchmark.evaluate(
    chunker=RecursiveChunker(chunk_size=512),
    embedding_model="voyage-3-large",
    k=[1, 5, 10],
)
print(result)
```

## 🧩 Available Benchmarks

### Full Benchmark

The full MTCB benchmark spans 9 domains with ~17k questions across ~3k documents:

| Dataset | Domain | Documents | Questions |
|---------|--------|----------:|----------:|
| [🧸 Gacha](https://huggingface.co/datasets/chonkie-ai/gacha) | Classic Literature (Gutenberg) | 100 | 2,878 |
| [💼 Ficha](https://huggingface.co/datasets/chonkie-ai/ficha) | SEC Financial Filings | 88 | 1,331 |
| [📝 Macha](https://huggingface.co/datasets/chonkie-ai/macha) | GitHub READMEs | 445 | 1,812 |
| [💻 Cocha](https://huggingface.co/datasets/chonkie-ai/cocha) | Multilingual Code | 1,000 | 2,372 |
| [📊 Tacha](https://huggingface.co/datasets/chonkie-ai/tacha) | Financial Tables (TAT-QA) | 349 | 2,065 |
| [🔬 Sencha](https://huggingface.co/datasets/chonkie-ai/sencha) | Scientific Papers (QASPER) | 243 | 1,507 |
| [⚖️ Hojicha](https://huggingface.co/datasets/chonkie-ai/hojicha) | Legal Contracts (CUAD) | 194 | 1,568 |
| [🏥 Ryokucha](https://huggingface.co/datasets/chonkie-ai/ryokucha) | Medical Guidelines (NICE/CDC/WHO) | 241 | 1,351 |
| [🎓 Genmaicha](https://huggingface.co/datasets/chonkie-ai/genmaicha) | MIT OCW Lecture Transcripts | 250 | 2,037 |
| | **Total** | **2,910** | **16,921** |

### Nano Benchmark

For fast iteration during development, MTCB provides a lightweight nano benchmark with ~100 questions per dataset. Documents are selected to maximize question density:

| Dataset | Domain | Documents | Questions |
|---------|--------|----------:|----------:|
| [🧸 nano-gacha](https://huggingface.co/datasets/chonkie-ai/nano-gacha) | Classic Literature | 5 | 100 |
| [💼 nano-ficha](https://huggingface.co/datasets/chonkie-ai/nano-ficha) | SEC Financial Filings | 5 | 100 |
| [📝 nano-macha](https://huggingface.co/datasets/chonkie-ai/nano-macha) | GitHub READMEs | 19 | 100 |
| [💻 nano-cocha](https://huggingface.co/datasets/chonkie-ai/nano-cocha) | Multilingual Code | 26 | 100 |
| [📊 nano-tacha](https://huggingface.co/datasets/chonkie-ai/nano-tacha) | Financial Tables | 11 | 100 |
| [🔬 nano-sencha](https://huggingface.co/datasets/chonkie-ai/nano-sencha) | Scientific Papers | 13 | 100 |
| [⚖️ nano-hojicha](https://huggingface.co/datasets/chonkie-ai/nano-hojicha) | Legal Contracts | 10 | 100 |
| [🏥 nano-ryokucha](https://huggingface.co/datasets/chonkie-ai/nano-ryokucha) | Medical Guidelines | 12 | 100 |
| [🎓 nano-genmaicha](https://huggingface.co/datasets/chonkie-ai/nano-genmaicha) | Lecture Transcripts | 7 | 100 |
| | **Total** | **108** | **900** |

## 🔧 Usage

MTCB works with [Chonkie](https://github.com/chonkie-inc/chonkie) — any chunker that extends `chonkie.BaseChunker` is supported out of the box.

### Full Benchmark

Run the complete benchmark across all 9 domains:

```python
from mtcb import Benchmark
from chonkie import RecursiveChunker

benchmark = Benchmark()
result = benchmark.evaluate(
    chunker=RecursiveChunker(chunk_size=512),
    embedding_model="voyage-3-large",
    k=[1, 5, 10],
)
print(result)
```

### Individual Evaluators

Run a single domain-specific evaluator:

```python
from mtcb import GachaEvaluator
from chonkie import RecursiveChunker

evaluator = GachaEvaluator(
    chunker=RecursiveChunker(chunk_size=1000),
    embedding_model="voyage-3-large",
    cache_dir="./cache"
)

result = evaluator.evaluate(k=[1, 3, 5, 10])
print(result)
```

### Custom Datasets

Evaluate on your own corpus using `SimpleEvaluator`:

```python
from mtcb import SimpleEvaluator
from chonkie import RecursiveChunker

evaluator = SimpleEvaluator(
    corpus=["Your document text here...", "Another document..."],
    questions=["What is X?", "How does Y work?"],
    relevant_passages=["passage that must be in retrieved chunk", "another passage"],
    chunker=RecursiveChunker(chunk_size=1000),
    embedding_model="voyage-3-large",
)

result = evaluator.evaluate(k=[1, 3, 5, 10])
print(result)
```

### Dataset Generation

Generate verified QA datasets from your own documents:

```python
from mtcb import DatasetGenerator

generator = DatasetGenerator(deduplicate=True)
result = generator.generate(
    corpus=["Your document text..."],
    samples_per_document=10,
    output_path="./output.jsonl",
)

print(f"Generated {result.total_verified} verified samples")
for sample in result.samples:
    print(f"Q: {sample.question}")
    print(f"A: {sample.answer}")
```

## 📊 Metrics

MTCB evaluates retrieval quality using:

- **Recall@k**: Percentage of questions where the relevant passage appears in the top-k results
- **Precision@k**: Ratio of relevant chunks in the top-k results
- **MRR@k**: Mean Reciprocal Rank — how high the first relevant result ranks
- **NDCG@k**: Normalized Discounted Cumulative Gain — position-weighted relevance scoring

## 📚 Citation

If you use MTCB in your research, please cite:

```bibtex
@software{mtcb2025,
  author = {Bhavnick Minhas and Shreyash Nigam},
  title = {MTCB: Massive Text Chunking Benchmark},
  url = {https://github.com/chonkie-inc/mtcb},
  version = {0.1.0},
  year = {2025},
}
```
