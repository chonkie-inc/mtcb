<div align="center">

# 🔬 mtcb ✨
_Massive Text Chunking Benchmark. Evaluate your RAG chunking with ease._

</div>

Chunking is a crucial step for RAG systems and LLM workflows. Most of the time, chunk quality is rarely evaluated even though it can have a significant impact on the performance of the system. MTCB makes it super easy to evaluate your chunks!

## 📦 Installation

Installation is super easy! Just run the following command in your terminal:

```bash
pip install mtcb
```

## 🧑‍⚖️ Usage

MTCB works together with [chonkie](https://github.com/chonkie-inc/chonkie) to evaluate your chunks. It supports all the chunkers that chonkie supports, and as long as any chunker can be wrapped in a `chonkie.BaseChunker` wrapper, MTCB will support it.

### Gacha Benchmark

The easiest way to evaluate your chunking strategy is with the Gacha benchmark:

```python
from mtcb import GachaEvaluator
from chonkie import RecursiveChunker

# Initialize the evaluator with your chunker
evaluator = GachaEvaluator(
    chunker=RecursiveChunker(chunk_size=1000),
    embedding_model="voyage-3-large",
    cache_dir="./cache"
)

# Evaluate your chunks
result = evaluator.evaluate(k=[1, 3, 5, 10])

# Print the results
print(result)
```

### Custom Datasets

You can also evaluate on your own datasets using `SimpleEvaluator`:

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

MTCB can also generate verified QA datasets from your documents for evaluation:

```python
from mtcb import DatasetGenerator

generator = DatasetGenerator(deduplicate=True)
result = generator.generate(
    corpus=["Your document text..."],
    samples_per_document=10,
    output_path="./output.jsonl",  # Save progress incrementally
)

print(f"Generated {result.total_verified} verified samples")
for sample in result.samples:
    print(f"Q: {sample.question}")
    print(f"A: {sample.answer}")
```

### Metrics

MTCB calculates:
- **Recall@k**: Percentage of questions where the relevant passage is in the top-k results
- **Precision@k**: Ratio of relevant chunks in the top-k results
- **MRR@k**: Mean Reciprocal Rank at k
- **NDCG@k**: Normalized Discounted Cumulative Gain at k

## 🧩 Available Benchmarks

### Full Benchmark

The full MTCB benchmark contains 16,974 questions across 3,202 documents spanning 9 diverse domains:

| Dataset | Domain | Documents | Questions |
|---------|--------|----------:|----------:|
| [🧸 Gacha](https://huggingface.co/datasets/chonkie-ai/gacha) | Classic Literature (Gutenberg) | 100 | 2,878 |
| [💼 Ficha](https://huggingface.co/datasets/chonkie-ai/ficha) | SEC Financial Filings | 88 | 1,331 |
| [📝 Macha](https://huggingface.co/datasets/chonkie-ai/macha) | GitHub READMEs | 445 | 1,812 |
| [💻 Cocha](https://huggingface.co/datasets/chonkie-ai/cocha) | Multilingual Code | 1,000 | 2,372 |
| [📊 Tacha](https://huggingface.co/datasets/chonkie-ai/tacha) | Financial Tables (TAT-QA) | 349 | 2,065 |
| [🔬 Sencha](https://huggingface.co/datasets/chonkie-ai/sencha) | Scientific Papers (QASPER) | 250 | 1,146 |
| [⚖️ Hojicha](https://huggingface.co/datasets/chonkie-ai/hojicha) | Legal Contracts (CUAD) | 479 | 1,982 |
| [🏥 Ryokucha](https://huggingface.co/datasets/chonkie-ai/ryokucha) | Medical Guidelines (NICE/CDC/WHO) | 241 | 1,351 |
| [🎓 Genmaicha](https://huggingface.co/datasets/chonkie-ai/genmaicha) | MIT OCW Lecture Transcripts | 250 | 2,037 |
| | **Total** | **3,202** | **16,974** |

### Nano Benchmark

For fast iteration and testing, MTCB provides a lightweight nano benchmark with ~100 questions per dataset. Documents are selected to maximize question density (fewest documents needed for 100 questions):

| Dataset | Domain | Documents | Questions |
|---------|--------|----------:|----------:|
| [🧸 nano-gacha](https://huggingface.co/datasets/chonkie-ai/nano-gacha) | Classic Literature | 4 | 100 |
| [💼 nano-ficha](https://huggingface.co/datasets/chonkie-ai/nano-ficha) | SEC Financial Filings | 3 | 100 |
| [📝 nano-macha](https://huggingface.co/datasets/chonkie-ai/nano-macha) | GitHub READMEs | 19 | 100 |
| [💻 nano-cocha](https://huggingface.co/datasets/chonkie-ai/nano-cocha) | Multilingual Code | 26 | 100 |
| [📊 nano-tacha](https://huggingface.co/datasets/chonkie-ai/nano-tacha) | Financial Tables | 11 | 100 |
| [🔬 nano-sencha](https://huggingface.co/datasets/chonkie-ai/nano-sencha) | Scientific Papers | 12 | 100 |
| [⚖️ nano-hojicha](https://huggingface.co/datasets/chonkie-ai/nano-hojicha) | Legal Contracts | 8 | 100 |
| [🏥 nano-ryokucha](https://huggingface.co/datasets/chonkie-ai/nano-ryokucha) | Medical Guidelines | 12 | 100 |
| [🎓 nano-genmaicha](https://huggingface.co/datasets/chonkie-ai/nano-genmaicha) | Lecture Transcripts | 7 | 100 |
| | **Total** | **102** | **900** |

Use `NanoBenchmark` for quick evaluations during development:

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


## 📚 Citation

If you use MTCB in your work, please cite it as follows:

```bibtex
@software{mtcb2025,
  author = {Bhavnick Minhas and Shreyash Nigam},
  title = {MTCB: Massive Text Chunking Benchmark},
  url = {https://github.com/chonkie-inc/mtcb},
  version = {0.1.0},
  year = {2025},
}
```
