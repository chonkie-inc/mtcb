<div align="center">

![🦉 Judie Logo](./assets/judie-logo.png)

# 🦉 judie ✨
_Judge and evaluate your chunk quality with Judie, the Owl! Quick, easy, and effective!_

</div>

Chunking is a crucial step for RAG systems and LLM workflows. Most of the time, chunk quality is rarely evaluated even though it can have a significant impact on the performance of the system. Judie makes it super easy to evaluate your chunks!

## 📦 Installation

Installation is super easy! Just run the following command in your terminal:

```bash
pip install judie
```

## 🧑‍⚖️ Usage

Judie works together with [chonkie](https://github.com/chonkie-inc/chonkie) to evaluate your chunks. It supports all the chunkers that chonkie supports, and as long as any chunker can be wrapped in a `chonkie.BaseChunker` wrapper, Judie will support it.

### Gacha Benchmark

The easiest way to evaluate your chunking strategy is with the Gacha benchmark:

```python
from judie import GachaEvaluator
from chonkie import RecursiveChunker

# Initialize the evaluator with your chunker
evaluator = GachaEvaluator(
    chunker=RecursiveChunker(chunk_size=1000),
    embedding_model="voyage-3-large",  # Uses Catsu for embeddings
    tokenizer="auto",  # Auto-detect tokenizer from model
    cache_dir="./cache"  # Cache embeddings for faster re-runs
)

# Evaluate your chunks
result = evaluator.evaluate(k=[1, 3, 5, 10])

# Print the results
print(result)
```

### Custom Datasets

You can also evaluate on your own datasets using `SimpleEvaluator`:

```python
from judie import SimpleEvaluator
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

### Supported Embedding Models

Judie uses [Catsu](https://github.com/chonkie-inc/catsu) under the hood, which supports:

- **Voyage AI**: `voyage-3-large`, `voyage-3`, `voyage-code-3`, etc.
- **OpenAI**: `text-embedding-3-large`, `text-embedding-3-small`
- **Cohere**: `embed-english-v3.0`, `embed-multilingual-v3.0`
- **Jina**: `jina-embeddings-v3`, `jina-embeddings-v2-base-en`

For backward compatibility, `model2vec://` prefixed models are also supported.

### Metrics

Judie calculates:
- **Recall@k**: Percentage of questions where the relevant passage is in the top-k results
- **MRR@k**: Mean Reciprocal Rank at k

## 🧩 Available Benchmarks

| Benchmark | Description |
|-----------|-------------|
| [🧸 Gacha](https://huggingface.co/datasets/chonkie-ai/gacha) | 🧸 Gacha is a corpus of 100 most popular textbooks from Gutenberg Corpus and numerous NIAH-style questions for evaluating chunking algorithms! |


## 📚 Citation

If you use Judie in your work, please cite it as follows:

```bibtex
@software{judie2025,
  author = {Bhavnick Minhas and Shreyash Nigam},
  title = {🦉 Judie: Evaluating chunk quality with LLM Judges},
  url = {https://github.com/chonkie-inc/judie},
  version = {0.1.0},
  year = {2025},
}
```
