#!/usr/bin/env python3
"""Generate QA pairs for Hojicha legal contracts dataset using DatasetGenerator with Groq Kimi K2.

This script regenerates the hojicha dataset with proper semantic questions
and chunk-must-contain passages that are suitable for retrieval benchmarking.

Usage:
    export GROQ_API_KEY=your_key
    python scripts/generate_hojicha_qa.py
    python scripts/generate_hojicha_qa.py --push  # Push to HuggingFace when done
"""

import argparse
import json
import os
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset

from mtcb import DatasetGenerator, DatasetPromptTemplate, ExactMatchValidator


# Custom prompt template for legal contracts
def get_legal_prompt_template() -> DatasetPromptTemplate:
    """Create a prompt template optimized for legal contract QA generation."""
    return DatasetPromptTemplate(
        name="legal",
        generation_template="""You are generating question-answer pairs for a legal contract RAG benchmark.

Given this legal contract excerpt, generate a question-answer pair that:
1. Asks a specific question about contract terms, parties, dates, obligations, or conditions
2. Can be answered by information in this excerpt
3. Is a natural question a lawyer or business person might ask

Requirements:
- question: Clear, specific question (NOT "highlight parts related to X")
- answer: Brief answer to the question
- chunk_must_contain: EXACT quote (150-400 chars) from the text containing the answer

IMPORTANT: chunk_must_contain must be VERBATIM from the text, not paraphrased.

Good question examples:
- "What is the effective date of this agreement?"
- "Who are the parties involved in this contract?"
- "What happens if either party breaches the agreement?"
- "What is the governing law for this contract?"

{existing_questions_section}

Text excerpt:
{chunk_text}

Output JSON with keys: question, answer, chunk_must_contain""",
        deduplication_template="""Analyze these legal contract questions for semantic similarity.
Consider questions duplicates if they ask about the same:
- Contract clause or provision
- Party or entity
- Date or deadline
- Obligation or right
- Legal term or condition

Questions:
{questions_json}

Return JSON with:
- "groups": list of groups with indices of similar questions
- "unique_indices": list of one representative index from each group""",
    )


def main():
    parser = argparse.ArgumentParser(description="Generate hojicha QA pairs")
    parser.add_argument(
        "--push", action="store_true", help="Push to HuggingFace when done"
    )
    parser.add_argument(
        "--max-docs", type=int, default=None, help="Max documents to process"
    )
    parser.add_argument(
        "--samples-per-doc",
        type=int,
        default=10,
        help="Samples per document (default: 10)",
    )
    args = parser.parse_args()

    # Check for API key
    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        print("Please set GROQ_API_KEY environment variable")
        sys.exit(1)

    # Load selected document indices
    selection_file = Path("scripts/hojicha_selected_docs.json")
    if not selection_file.exists():
        print("Please run document selection first")
        sys.exit(1)

    with open(selection_file) as f:
        selection = json.load(f)

    selected_indices = set(selection["indices"])
    print(f"Selected {len(selected_indices)} documents for QA generation")

    # Load corpus
    corpus = load_dataset("chonkie-ai/hojicha", "corpus", split="train")

    # Filter to selected documents and prepare corpus list
    selected_docs = []
    doc_titles = []
    for i, doc in enumerate(corpus):
        if i in selected_indices:
            selected_docs.append(doc["text"])
            doc_titles.append(doc["title"])

    print(f"Loaded {len(selected_docs)} selected documents")

    if args.max_docs:
        selected_docs = selected_docs[: args.max_docs]
        doc_titles = doc_titles[: args.max_docs]
        print(f"Limited to {len(selected_docs)} documents")

    # Set up genie with Groq (OpenAI-compatible API)
    from chonkie import OpenAIGenie, RecursiveChunker

    genie = OpenAIGenie(
        model="moonshotai/kimi-k2-instruct",
        base_url="https://api.groq.com/openai/v1",
        api_key=groq_key,
    )
    print("Using Groq API with Kimi K2 model")

    # Set up chunker with larger chunks for legal documents
    chunker = RecursiveChunker(chunk_size=2048)

    # Set up generator
    generator = DatasetGenerator(
        genie=genie,
        chunker=chunker,
        validator=ExactMatchValidator(),
        prompt_template=get_legal_prompt_template(),
        deduplicate=True,
        show_progress_bar=True,
    )

    # Output path for incremental saving
    output_path = Path("scripts/hojicha_generation_progress.jsonl")

    print(f"\nGenerating {args.samples_per_doc} samples per document...")
    print(f"Total target: {len(selected_docs) * args.samples_per_doc} samples")

    # Generate
    result = generator.generate(
        corpus=selected_docs,
        samples_per_document=args.samples_per_doc,
        max_retries=3,
        output_path=output_path,
    )

    # Print summary
    print(f"\n{'='*60}")
    print("GENERATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total generated: {result.total_generated}")
    print(f"Total verified: {result.total_verified}")
    print(f"Failed validation: {result.failed_validation_count}")
    print(f"Duplicates removed: {result.duplicate_count}")
    print(f"Time: {result.generation_time_seconds:.1f}s")

    if args.push and result.samples:
        print(f"\nPushing to HuggingFace...")

        # Map document_id back to title
        questions_data = []
        for sample in result.samples:
            doc_id = sample.document_id
            if doc_id < len(doc_titles):
                questions_data.append(
                    {
                        "question": sample.question,
                        "document_title": doc_titles[doc_id],
                        "chunk-must-contain": sample.chunk_must_contain,
                    }
                )

        # Get unique document IDs that have valid questions
        valid_doc_ids = set(s.document_id for s in result.samples)

        # Prepare corpus dataset (only docs with valid questions)
        corpus_data = []
        for doc_id in valid_doc_ids:
            if doc_id < len(selected_docs):
                corpus_data.append(
                    {
                        "title": doc_titles[doc_id],
                        "text": selected_docs[doc_id],
                    }
                )

        print(f"Corpus: {len(corpus_data)} documents")
        print(f"Questions: {len(questions_data)}")

        # Push to HuggingFace
        corpus_ds = Dataset.from_list(corpus_data)
        questions_ds = Dataset.from_list(questions_data)

        DatasetDict({"train": corpus_ds}).push_to_hub(
            "chonkie-ai/hojicha", config_name="corpus"
        )
        DatasetDict({"train": questions_ds}).push_to_hub(
            "chonkie-ai/hojicha", config_name="questions"
        )

        print("Successfully pushed to chonkie-ai/hojicha")

    # Also save final results locally
    final_output = Path("scripts/hojicha_questions_final.json")
    with open(final_output, "w") as f:
        json.dump(
            {
                "total_verified": result.total_verified,
                "samples": [
                    {
                        "question": s.question,
                        "answer": s.answer,
                        "chunk_must_contain": s.chunk_must_contain,
                        "document_id": s.document_id,
                        "document_title": doc_titles[s.document_id]
                        if s.document_id < len(doc_titles)
                        else None,
                    }
                    for s in result.samples
                ],
            },
            f,
            indent=2,
        )
    print(f"Saved final results to {final_output}")


if __name__ == "__main__":
    main()
