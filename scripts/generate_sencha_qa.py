#!/usr/bin/env python3
"""Generate QA pairs for Sencha scientific papers dataset using DatasetGenerator with Groq Kimi K2.

This script regenerates the sencha dataset with high-quality, paper-specific questions
that avoid generic templates like "What is the dataset size?" and instead ask about
specific methods, findings, and contributions unique to each paper.

Usage:
    export GROQ_API_KEY=your_key
    python scripts/generate_sencha_qa.py
    python scripts/generate_sencha_qa.py --push  # Push to HuggingFace when done
"""

import argparse
import json
import os
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset

from mtcb import DatasetGenerator, DatasetPromptTemplate, ExactMatchValidator


def get_scientific_prompt_template() -> DatasetPromptTemplate:
    """Create a prompt template optimized for scientific paper QA generation."""
    return DatasetPromptTemplate(
        name="scientific",
        generation_template="""You are generating question-answer pairs for a scientific paper RAG benchmark.

Given this excerpt from a scientific paper, generate a question-answer pair that:
1. Asks about SPECIFIC details unique to THIS paper (methods, results, datasets, models)
2. Includes enough context in the question to distinguish it from similar papers
3. Is a natural question a researcher might ask when searching for this specific work

IMPORTANT - Avoid generic questions like:
- "What is the dataset size?" (too generic)
- "What baseline is used?" (too generic)
- "What are the results?" (too vague)

Instead, ask specific questions like:
- "What preprocessing steps are used for the Twitter hate speech dataset in the multimodal detection study?"
- "How does the BERT-based model handle code-switching in the Hindi-English sentiment analysis?"
- "What F1 score does the graph neural network achieve on the fake news detection task?"

Requirements:
- question: Specific question with context (mention the paper's domain/method if relevant)
- answer: Brief, factual answer
- chunk_must_contain: EXACT quote (200-600 chars) from the text containing the answer

{existing_questions_section}

Text excerpt:
{chunk_text}

Output JSON with keys: question, answer, chunk_must_contain""",
        deduplication_template="""Analyze these scientific questions for semantic similarity.
Consider questions duplicates if they:
- Ask about the same method, result, or finding
- Reference the same experiment or dataset
- Would have substantially overlapping answers

Questions:
{questions_json}

Return JSON with:
- "groups": list of groups with indices of similar questions
- "unique_indices": list of one representative index from each group""",
    )


def main():
    parser = argparse.ArgumentParser(description="Generate sencha QA pairs")
    parser.add_argument(
        "--push", action="store_true", help="Push to HuggingFace when done"
    )
    parser.add_argument(
        "--max-docs", type=int, default=None, help="Max documents to process"
    )
    parser.add_argument(
        "--samples-per-doc",
        type=int,
        default=8,
        help="Samples per document (default: 8)",
    )
    args = parser.parse_args()

    # Check for API key
    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        print("Please set GROQ_API_KEY environment variable")
        sys.exit(1)

    # Load corpus
    print("Loading sencha corpus...")
    corpus = load_dataset("chonkie-ai/sencha", "corpus", split="train")
    print(f"Loaded {len(corpus)} papers")

    # Prepare corpus list
    doc_texts = []
    doc_ids = []
    doc_titles = []
    for doc in corpus:
        doc_texts.append(doc["text"])
        doc_ids.append(doc["id"])
        doc_titles.append(doc["title"])

    if args.max_docs:
        doc_texts = doc_texts[: args.max_docs]
        doc_ids = doc_ids[: args.max_docs]
        doc_titles = doc_titles[: args.max_docs]
        print(f"Limited to {len(doc_texts)} documents")

    # Set up genie with Groq
    from chonkie import OpenAIGenie, RecursiveChunker

    genie = OpenAIGenie(
        model="moonshotai/kimi-k2-instruct",
        base_url="https://api.groq.com/openai/v1",
        api_key=groq_key,
    )
    print("Using Groq API with Kimi K2 model")

    # Set up chunker - larger chunks for scientific content
    chunker = RecursiveChunker(chunk_size=2048)

    # Set up generator
    generator = DatasetGenerator(
        genie=genie,
        chunker=chunker,
        validator=ExactMatchValidator(),
        prompt_template=get_scientific_prompt_template(),
        deduplicate=True,
        show_progress_bar=True,
    )

    # Output path for incremental saving
    output_path = Path("scripts/sencha_generation_progress.jsonl")

    print(f"\nGenerating {args.samples_per_doc} samples per document...")
    print(f"Total target: {len(doc_texts) * args.samples_per_doc} samples")

    # Generate
    result = generator.generate(
        corpus=doc_texts,
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

        # Map document_id back to paper_id and title
        questions_data = []
        for sample in result.samples:
            doc_id = sample.document_id
            if doc_id < len(doc_ids):
                questions_data.append(
                    {
                        "id": f"sencha_q_{len(questions_data)}",
                        "paper_id": doc_ids[doc_id],
                        "question": sample.question,
                        "answer": sample.answer,
                        "chunk-must-contain": sample.chunk_must_contain,
                    }
                )

        # Get unique document IDs that have valid questions
        valid_doc_ids = set(s.document_id for s in result.samples)

        # Prepare corpus dataset (only docs with valid questions)
        corpus_data = []
        for doc_id in sorted(valid_doc_ids):
            if doc_id < len(doc_texts):
                corpus_data.append(
                    {
                        "id": doc_ids[doc_id],
                        "title": doc_titles[doc_id],
                        "text": doc_texts[doc_id],
                        "num_sections": corpus[doc_id]["num_sections"],
                    }
                )

        print(f"Corpus: {len(corpus_data)} papers")
        print(f"Questions: {len(questions_data)}")

        # Push to HuggingFace
        corpus_ds = Dataset.from_list(corpus_data)
        questions_ds = Dataset.from_list(questions_data)

        DatasetDict({"train": corpus_ds}).push_to_hub(
            "chonkie-ai/sencha", config_name="corpus"
        )
        DatasetDict({"train": questions_ds}).push_to_hub(
            "chonkie-ai/sencha", config_name="questions"
        )

        print("Successfully pushed to chonkie-ai/sencha")

    # Also save final results locally
    final_output = Path("scripts/sencha_questions_final.json")
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
                        "paper_id": doc_ids[s.document_id]
                        if s.document_id < len(doc_ids)
                        else None,
                        "paper_title": doc_titles[s.document_id]
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
