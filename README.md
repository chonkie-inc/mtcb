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

Here's a simple example of how to use Judie:

```python 
from judie import Evaluator 
from chonkie import RecursiveChunker

chunker = RecursiveChunker(chunk_size=1000, chunk_overlap=200)

# Load your dataset
dataset = ["...", "..."]

# Initialize the evaluator
evaluator = Evaluator(chunker=chunker,
                      dataset=dataset, 
                      embedding_model=embedding_model)

# Evaluate your chunks
metrics = evaluator.evaluate()

# Print the metrics
print(metrics['Recall@10'])
```

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
