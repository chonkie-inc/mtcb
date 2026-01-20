# Plan: Verified Dataset Generation for MTCB + New Genies for Chonkie

## Summary

Replace the existing `DatasetGenerator` with a new implementation that generates verified QA datasets with LLM-based semantic deduplication, following the patterns from the Ficha pipeline.

## Approach

**Replace `generator.py`** with a new `dataset/` module containing composable components:
- Clean separation of concerns
- Components can be used standalone or composed
- Validators and deduplicators are swappable

## File Structure

```
src/mtcb/
├── __init__.py                 # Update exports
├── generator.py                # REMOVE (replaced by dataset/)
└── dataset/                    # NEW MODULE
    ├── __init__.py             # Module exports
    ├── types.py                # Pydantic models (GeneratedSample, GenerationResult)
    ├── prompts.py              # Prompt templates (configurable by domain)
    ├── validator.py            # ExactMatchValidator, FuzzyMatchValidator
    ├── deduplicator.py         # LLMDeduplicator, EmbeddingDeduplicator
    └── generator.py            # DatasetGenerator main class
```

## Implementation Details

### 1. `dataset/types.py`

```python
class GeneratedSample(BaseModel):
    question: str
    answer: str
    chunk_must_contain: str  # Must exist verbatim in source chunk
    document_id: Optional[int] = None
    chunk_id: Optional[str] = None
    verified: bool = False

@dataclass
class GenerationResult:
    samples: List[GeneratedSample]
    total_generated: int
    total_verified: int
    failed_validation_count: int
    duplicate_count: int
    generation_time_seconds: float
```

### 2. `dataset/prompts.py`

- `DatasetPromptTemplate` dataclass with `generation_template` and `deduplication_template`
- `@classmethod` factories: `default()`, `strict()`, `financial()`, etc.
- Templates based on Ficha's detailed instructions

### 3. `dataset/validator.py`

```python
class ExactMatchValidator:
    def validate(self, sample: GeneratedSample, chunk_text: str) -> bool:
        return sample.chunk_must_contain in chunk_text  # Critical check

class FuzzyMatchValidator:
    def __init__(self, similarity_threshold: float = 0.95): ...
```

### 4. `dataset/deduplicator.py`

```python
class LLMDeduplicator:
    def __init__(self, genie, batch_size=50, max_passes=3): ...
    def deduplicate(self, samples: List) -> List:
        # Batch processing with multi-pass

class EmbeddingDeduplicator:
    def __init__(self, embedding_model, similarity_threshold=0.92): ...
```

### 5. `dataset/generator.py`

```python
class DatasetGenerator:
    def __init__(
        self,
        genie: Optional[BaseGenie] = None,          # Default: GeminiGenie
        chunker: Optional[BaseChunker] = None,      # Default: RecursiveChunker
        validator: Optional[SampleValidator] = None,
        deduplicator: Optional[SampleDeduplicator] = None,
        prompt_template: Optional[DatasetPromptTemplate] = None,
        deduplicate: bool = True,
        show_progress_bar: bool = True,
    ): ...

    def generate(
        self,
        corpus: Union[str, List[str]],
        samples_per_document: int = 10,
        max_retries: int = 3,
    ) -> GenerationResult: ...

    def to_dataset_dict(self, result, corpus) -> DatasetDict: ...
```

### 6. Update `__init__.py`

Export: `DatasetGenerator`, `GeneratedSample`, `GenerationResult`, `DatasetPromptTemplate`, `ExactMatchValidator`, `LLMDeduplicator`

## Example Usage

```python
from mtcb import DatasetGenerator

# Simple (uses GeminiGenie by default)
generator = DatasetGenerator(deduplicate=True)
result = generator.generate(corpus=["..."], samples_per_document=10)

# Advanced - custom genie and settings
from mtcb.dataset import LLMDeduplicator, DatasetPromptTemplate
from chonkie import GeminiGenie

generator = DatasetGenerator(
    genie=GeminiGenie(model="gemini-2.0-flash"),
    prompt_template=DatasetPromptTemplate.strict(),
    deduplicator=LLMDeduplicator(batch_size=100),
)

# With Groq (after Part 2 is implemented)
from chonkie import GroqGenie
generator = DatasetGenerator(genie=GroqGenie(model="llama-3.3-70b-versatile"))

# With Cerebras (after Part 2 is implemented)
from chonkie import CerebrasGenie
generator = DatasetGenerator(genie=CerebrasGenie(model="llama-3.3-70b"))
```

## Critical Files to Modify

| File | Action |
|------|--------|
| `src/mtcb/generator.py` | Remove |
| `src/mtcb/dataset/__init__.py` | Create (exports) |
| `src/mtcb/dataset/types.py` | Create |
| `src/mtcb/dataset/prompts.py` | Create |
| `src/mtcb/dataset/validator.py` | Create |
| `src/mtcb/dataset/deduplicator.py` | Create |
| `src/mtcb/dataset/generator.py` | Create |
| `src/mtcb/__init__.py` | Update (replace exports) |

## Key Design Decisions

1. **Exact match validation is mandatory** - `chunk_must_contain in chunk_text` check prevents hallucinated passages
2. **LLM deduplication by default** - More accurate than embedding-based for semantic similarity
3. **Batch deduplication** (50 items) with multi-pass - Memory efficient for large sets
4. **Protocol-based validators/deduplicators** - Easy to extend with custom implementations

## Verification

After implementation, test with:

```bash
# Activate venv
source .venv/bin/activate

# Run basic test
python -c "
from mtcb import DatasetGenerator
gen = DatasetGenerator()
result = gen.generate(['This is a test document about Python programming.'], samples_per_document=2)
print(f'Generated: {result.total_generated}, Verified: {result.total_verified}')
for s in result.samples:
    print(f'Q: {s.question}')
    print(f'A: {s.answer}')
    print(f'Verified: {s.verified}')
"

# Run pytest if tests exist
pytest -n auto tests/
```

---

# Part 2: New Genies for Chonkie

## Summary

Add `GroqGenie` and `CerebrasGenie` to chonkie. Both providers have OpenAI-compatible APIs, making implementation straightforward.

## File Structure

```
src/chonkie/genie/
├── __init__.py          # Update exports
├── base.py              # BaseGenie (existing)
├── gemini.py            # GeminiGenie (existing)
├── openai.py            # OpenAIGenie (existing)
├── azure_openai.py      # AzureOpenAIGenie (existing)
├── groq.py              # NEW: GroqGenie
└── cerebras.py          # NEW: CerebrasGenie
```

## Implementation Details

### 1. `genie/groq.py`

```python
class GroqGenie(BaseGenie):
    """Groq's Genie - fast inference on Groq hardware."""

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        api_key: Optional[str] = None,  # Falls back to GROQ_API_KEY
    ):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key)
        self.model = model

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    def generate_json(self, prompt: str, schema: "BaseModel") -> dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)

    @classmethod
    def _is_available(cls) -> bool:
        return importutil.find_spec("groq") is not None
```

**Notes:**
- Uses `groq` Python SDK
- Default model: `llama-3.3-70b-versatile` (fast, capable)
- JSON mode via `response_format={"type": "json_object"}`

### 2. `genie/cerebras.py`

```python
class CerebrasGenie(BaseGenie):
    """Cerebras Genie - fastest inference on Cerebras hardware."""

    def __init__(
        self,
        model: str = "llama-3.3-70b",
        api_key: Optional[str] = None,  # Falls back to CEREBRAS_API_KEY
    ):
        self.api_key = api_key or os.environ.get("CEREBRAS_API_KEY")
        self.client = Cerebras(api_key=self.api_key)
        self.model = model

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    def generate_json(self, prompt: str, schema: "BaseModel") -> dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)

    @classmethod
    def _is_available(cls) -> bool:
        return importutil.find_spec("cerebras") is not None
```

**Notes:**
- Uses `cerebras-cloud-sdk` Python SDK
- Default model: `llama-3.3-70b`
- OpenAI-compatible API

### 3. Update `genie/__init__.py`

```python
from .base import BaseGenie
from .gemini import GeminiGenie
from .openai import OpenAIGenie
from .azure_openai import AzureOpenAIGenie
from .groq import GroqGenie
from .cerebras import CerebrasGenie

__all__ = [
    "BaseGenie",
    "GeminiGenie",
    "OpenAIGenie",
    "AzureOpenAIGenie",
    "GroqGenie",
    "CerebrasGenie",
]
```

### 4. Update `chonkie/__init__.py`

Add `GroqGenie` and `CerebrasGenie` to the main exports.

### 5. Update `pyproject.toml`

Add optional dependencies:

```toml
[project.optional-dependencies]
groq = ["groq>=0.4.0"]
cerebras = ["cerebras-cloud-sdk>=1.0.0"]
```

## Critical Files to Modify (Chonkie)

| File | Action |
|------|--------|
| `src/chonkie/genie/groq.py` | Create |
| `src/chonkie/genie/cerebras.py` | Create |
| `src/chonkie/genie/__init__.py` | Update (add exports) |
| `src/chonkie/__init__.py` | Update (add exports) |
| `pyproject.toml` | Update (add optional deps) |

## Verification

```bash
cd /Users/bhavnick/Workspace/chonkie-main
source .venv/bin/activate

# Test Groq
python -c "
from chonkie import GroqGenie
genie = GroqGenie()
print(genie.generate('Say hello'))
"

# Test Cerebras
python -c "
from chonkie import CerebrasGenie
genie = CerebrasGenie()
print(genie.generate('Say hello'))
"
```
