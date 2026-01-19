"""Dataset generator for RAG evaluation."""

import json
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING, Union

from datasets import Dataset, DatasetDict
from tqdm import tqdm

from .deduplicator import LLMDeduplicator, NoOpDeduplicator, SampleDeduplicator
from .prompts import DatasetPromptTemplate
from .types import GeneratedSample, GenerationResult
from .validator import ExactMatchValidator, SampleValidator

if TYPE_CHECKING:
    from chonkie import BaseChunker, BaseGenie


class DatasetGenerator:
    """Generate verified QA datasets from documents with deduplication.

    Features:
    - Configurable LLM (genie) and chunker
    - Source text validation (chunk_must_contain)
    - LLM-based semantic deduplication
    - Customizable prompt templates
    - Incremental saving with resume support via output_path
    - Returns both GenerationResult and DatasetDict formats
    """

    def __init__(
        self,
        genie: "BaseGenie | None" = None,
        chunker: "BaseChunker | None" = None,
        validator: SampleValidator | None = None,
        deduplicator: SampleDeduplicator | None = None,
        prompt_template: DatasetPromptTemplate | None = None,
        deduplicate: bool = True,
        show_progress_bar: bool = True,
    ):
        """Initialize the DatasetGenerator.

        Args:
            genie: LLM genie for generation (defaults to GeminiGenie).
            chunker: Chunker for splitting documents (defaults to RecursiveChunker).
            validator: Sample validator (defaults to ExactMatchValidator).
            deduplicator: Deduplicator for removing duplicates (defaults to LLMDeduplicator).
            prompt_template: Prompt template for generation (defaults to DatasetPromptTemplate.default()).
            deduplicate: Whether to deduplicate samples.
            show_progress_bar: Whether to show progress during generation.
        """
        # Lazy import to avoid circular imports
        from chonkie import GeminiGenie, RecursiveChunker

        self.genie = genie or GeminiGenie(model="gemini-2.0-flash")
        self.chunker = chunker or RecursiveChunker()
        self.validator = validator or ExactMatchValidator()
        self.prompt_template = prompt_template or DatasetPromptTemplate.default()
        self.deduplicate = deduplicate
        self.show_progress_bar = show_progress_bar

        # Set up deduplicator
        if deduplicate:
            self.deduplicator = deduplicator or LLMDeduplicator(genie=self.genie)
        else:
            self.deduplicator = NoOpDeduplicator()

    def generate(
        self,
        corpus: Union[str, list[str]],
        samples_per_document: int = 10,
        max_retries: int = 3,
        output_path: Union[str, Path, None] = None,
    ) -> GenerationResult:
        """Generate QA samples from a corpus of documents.

        Args:
            corpus: Single document or list of documents.
            samples_per_document: Target number of samples per document.
            max_retries: Maximum retries for validation failures.
            output_path: Path to save intermediate results (JSONL format).
                If file exists, resumes from previous progress.

        Returns:
            GenerationResult with samples and statistics.
        """
        start_time = time.time()

        if isinstance(corpus, str):
            corpus = [corpus]

        # Load existing progress if output_path provided and file exists
        completed_docs: dict[int, list[GeneratedSample]] = {}
        if output_path:
            output_path = Path(output_path)
            completed_docs = self._load_progress(output_path)
            if completed_docs:
                print(f"Resuming: found {len(completed_docs)} completed documents")

        all_samples: list[GeneratedSample] = []
        total_generated = 0
        failed_validation = 0

        # Add samples from previously completed documents
        for doc_id in sorted(completed_docs.keys()):
            all_samples.extend(completed_docs[doc_id])

        # Calculate remaining work
        remaining_docs = [
            (doc_id, doc)
            for doc_id, doc in enumerate(corpus)
            if doc_id not in completed_docs
        ]
        total_target = len(remaining_docs) * samples_per_document

        progress_bar = None
        if self.show_progress_bar and total_target > 0:
            progress_bar = tqdm(
                total=total_target, desc="Generating samples", unit="samples"
            )

        try:
            for doc_id, document in remaining_docs:
                doc_samples, doc_generated, doc_failed = self._process_document(
                    document=document,
                    doc_id=doc_id,
                    samples_per_doc=samples_per_document,
                    max_retries=max_retries,
                    progress_bar=progress_bar,
                )
                all_samples.extend(doc_samples)
                total_generated += doc_generated
                failed_validation += doc_failed

                # Save progress after each document
                if output_path:
                    self._save_document_progress(output_path, doc_id, doc_samples)

        finally:
            if progress_bar:
                progress_bar.close()

        # Deduplicate
        pre_dedup_count = len(all_samples)
        if self.deduplicate and all_samples:
            all_samples = self.deduplicator.deduplicate(all_samples)
        duplicate_count = pre_dedup_count - len(all_samples)

        # Mark all remaining samples as verified
        for sample in all_samples:
            sample.verified = True

        generation_time = time.time() - start_time

        return GenerationResult(
            samples=all_samples,
            total_generated=total_generated,
            total_verified=len(all_samples),
            failed_validation_count=failed_validation,
            duplicate_count=duplicate_count,
            generation_time_seconds=generation_time,
        )

    def _load_progress(self, output_path: Path) -> dict[int, list[GeneratedSample]]:
        """Load previously completed documents from output file."""
        completed: dict[int, list[GeneratedSample]] = {}

        if not output_path.exists():
            return completed

        try:
            with open(output_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    doc_id = data["document_id"]
                    samples = [
                        GeneratedSample(**sample) for sample in data["samples"]
                    ]
                    completed[doc_id] = samples
        except (json.JSONDecodeError, KeyError):
            # If file is corrupted, start fresh
            return {}

        return completed

    def _save_document_progress(
        self, output_path: Path, doc_id: int, samples: list[GeneratedSample]
    ) -> None:
        """Append completed document samples to output file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "document_id": doc_id,
            "samples": [sample.model_dump() for sample in samples],
        }

        with open(output_path, "a") as f:
            f.write(json.dumps(data) + "\n")

    def _process_document(
        self,
        document: str,
        doc_id: int,
        samples_per_doc: int,
        max_retries: int,
        progress_bar=None,
    ) -> tuple[list[GeneratedSample], int, int]:
        """Process a single document and generate samples.

        Returns:
            Tuple of (samples, total_generated, failed_count)
        """
        chunks = self.chunker.chunk(document)

        if not chunks:
            if progress_bar:
                progress_bar.update(samples_per_doc)
            return [], 0, 0

        # Sample chunks intelligently
        sampled_chunks = self._sample_chunks(chunks, samples_per_doc)
        distribution = self._distribute_samples(sampled_chunks, samples_per_doc)

        samples: list[GeneratedSample] = []
        total_generated = 0
        failed_count = 0
        existing_questions: list[str] = []

        for chunk_idx, num_samples in distribution.items():
            chunk = sampled_chunks[chunk_idx]
            chunk_text = chunk.text if hasattr(chunk, "text") else str(chunk)
            chunk_id = f"doc{doc_id}_chunk{chunk_idx}"

            for _ in range(num_samples):
                sample = self._generate_sample(
                    chunk_text=chunk_text,
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    max_retries=max_retries,
                    existing_questions=existing_questions,
                )

                total_generated += 1

                if sample:
                    samples.append(sample)
                    existing_questions.append(sample.question)
                else:
                    failed_count += 1

                if progress_bar:
                    progress_bar.update(1)

        return samples, total_generated, failed_count

    def _sample_chunks(self, chunks: list, samples_per_doc: int) -> list:
        """Intelligently sample chunks based on sample count."""
        if not chunks:
            return []

        num_chunks = len(chunks)

        # If we need fewer samples than chunks, sample that many chunks
        if samples_per_doc >= num_chunks:
            return chunks

        # Weight by chunk length (prefer longer chunks)
        weights = []
        for chunk in chunks:
            text = chunk.text if hasattr(chunk, "text") else str(chunk)
            weights.append(len(text.split()))

        if sum(weights) == 0:
            return random.sample(chunks, samples_per_doc)

        # Weighted sampling
        sampled_indices = random.choices(
            range(len(chunks)), weights=weights, k=samples_per_doc
        )

        # Remove duplicates
        seen = set()
        sampled = []
        for idx in sampled_indices:
            if idx not in seen:
                sampled.append(chunks[idx])
                seen.add(idx)

        # Fill remaining if needed
        if len(sampled) < samples_per_doc:
            remaining = [i for i in range(len(chunks)) if i not in seen]
            for idx in remaining[: samples_per_doc - len(sampled)]:
                sampled.append(chunks[idx])

        return sampled

    def _distribute_samples(
        self, chunks: list, total_samples: int
    ) -> dict[int, int]:
        """Distribute samples across chunks."""
        if not chunks:
            return {}

        num_chunks = len(chunks)

        # One sample per chunk if enough chunks
        if total_samples <= num_chunks:
            return {i: 1 for i in range(total_samples)}

        # Distribute by chunk length
        weights = []
        for chunk in chunks:
            text = chunk.text if hasattr(chunk, "text") else str(chunk)
            weights.append(len(text.split()))

        total_weight = sum(weights)
        if total_weight == 0:
            # Equal distribution
            per_chunk = total_samples // num_chunks
            remainder = total_samples % num_chunks
            dist = {i: per_chunk for i in range(num_chunks)}
            for i in range(remainder):
                dist[i] += 1
            return dist

        # Weighted distribution
        dist = {}
        remaining = total_samples

        for i, weight in enumerate(weights[:-1]):
            count = max(1, int((weight / total_weight) * total_samples))
            dist[i] = min(count, remaining)
            remaining -= dist[i]

        dist[num_chunks - 1] = max(1, remaining)

        return dist

    def _generate_sample(
        self,
        chunk_text: str,
        doc_id: int,
        chunk_id: str,
        max_retries: int,
        existing_questions: list[str],
    ) -> GeneratedSample | None:
        """Generate a single sample with validation."""
        for _ in range(max_retries):
            try:
                prompt = self.prompt_template.format_generation_prompt(
                    chunk_text=chunk_text,
                    existing_questions=existing_questions,
                )

                result = self.genie.generate_json(prompt, GeneratedSample)

                # Create sample
                sample = GeneratedSample(
                    question=result.get("question", ""),
                    answer=result.get("answer", ""),
                    chunk_must_contain=result.get("chunk_must_contain", ""),
                    document_id=doc_id,
                    chunk_id=chunk_id,
                    verified=False,
                )

                # Validate
                if self.validator.validate(sample, chunk_text):
                    return sample

            except Exception:
                continue

        return None

    def to_dataset_dict(
        self, result: GenerationResult, corpus: Union[str, list[str]]
    ) -> DatasetDict:
        """Convert GenerationResult to HuggingFace DatasetDict.

        Args:
            result: The generation result.
            corpus: The original corpus documents.

        Returns:
            DatasetDict with 'corpus' and 'qa' splits.
        """
        if isinstance(corpus, str):
            corpus = [corpus]

        # Corpus dataset
        corpus_data = [
            {"document_id": i, "document": doc} for i, doc in enumerate(corpus)
        ]

        # QA dataset
        qa_data = [
            {
                "document_id": s.document_id,
                "question": s.question,
                "answer": s.answer,
                "chunk_must_contain": s.chunk_must_contain,
            }
            for s in result.samples
        ]

        return DatasetDict(
            {
                "corpus": Dataset.from_list(corpus_data),
                "qa": Dataset.from_list(qa_data),
            }
        )

    def __call__(
        self,
        corpus: Union[str, list[str]],
        samples_per_document: int = 10,
        max_retries: int = 3,
        output_path: Union[str, Path, None] = None,
    ) -> GenerationResult:
        """Callable interface - delegates to generate()."""
        return self.generate(corpus, samples_per_document, max_retries, output_path)

    def __repr__(self) -> str:
        return (
            f"DatasetGenerator("
            f"genie={self.genie}, "
            f"chunker={self.chunker}, "
            f"deduplicate={self.deduplicate})"
        )
