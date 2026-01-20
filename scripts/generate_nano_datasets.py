#!/usr/bin/env python3
"""Generate nano versions of MTCB datasets for fast iteration and testing.

This script creates minimal nano datasets by:
1. Mapping each question to its containing document
2. Ranking documents by question count (most questions first)
3. Selecting the fewest documents needed to get ~100 questions
4. Sampling exactly 100 questions from those documents

This minimizes document count and chunk count for fast benchmarking.

Usage:
    python scripts/generate_nano_datasets.py --push  # Generate and push to HuggingFace
    python scripts/generate_nano_datasets.py         # Generate locally only (dry run)
"""

import argparse
import random
from collections import defaultdict
from typing import Dict, List, Tuple

from datasets import Dataset, DatasetDict, load_dataset


# Dataset configurations: (hf_name, corpus_text_field, question_fields)
DATASETS = {
    "gacha": {
        "hf_name": "chonkie-ai/gacha",
        "corpus_text_field": "text",
        "question_field": "question",
        "passage_field": "chunk-must-contain",
    },
    "ficha": {
        "hf_name": "chonkie-ai/ficha",
        "corpus_text_field": "text",
        "question_field": "question",
        "passage_field": "chunk-must-contain",
    },
    "macha": {
        "hf_name": "chonkie-ai/macha",
        "corpus_text_field": "text",
        "question_field": "question",
        "passage_field": "chunk-must-contain",
    },
    "cocha": {
        "hf_name": "chonkie-ai/cocha",
        "corpus_text_field": "text",  # Some use "content"
        "question_field": "question",
        "passage_field": "chunk-must-contain",
    },
    "tacha": {
        "hf_name": "chonkie-ai/tacha",
        "corpus_text_field": "text",
        "question_field": "question",
        "passage_field": "chunk-must-contain",
    },
    "sencha": {
        "hf_name": "chonkie-ai/sencha",
        "corpus_text_field": "text",
        "question_field": "question",
        "passage_field": "chunk-must-contain",
    },
    "hojicha": {
        "hf_name": "chonkie-ai/hojicha",
        "corpus_text_field": "text",
        "question_field": "question",
        "passage_field": "chunk-must-contain",
    },
    "ryokucha": {
        "hf_name": "chonkie-ai/ryokucha",
        "corpus_text_field": "text",
        "question_field": "question",
        "passage_field": "chunk-must-contain",
    },
    "genmaicha": {
        "hf_name": "chonkie-ai/genmaicha",
        "corpus_text_field": "text",
        "question_field": "question",
        "passage_field": "chunk-must-contain",
    },
}


def get_text_from_doc(doc: Dict, text_field: str) -> str:
    """Extract text from document, handling field variations."""
    return doc.get(text_field, doc.get("content", ""))


def get_passage_from_question(question: Dict, passage_field: str) -> str:
    """Extract passage from question, handling field variations."""
    return question.get(passage_field, question.get("supporting_passage", ""))


def generate_nano_dataset(
    dataset_name: str,
    config: Dict,
    num_questions: int = 100,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """Generate a nano version of a dataset with minimal documents.

    Strategy:
    1. Map each question to its containing document
    2. Rank documents by question count (most questions first)
    3. Select fewest documents needed to get ≥num_questions
    4. Sample exactly num_questions from those documents

    Args:
        dataset_name: Name of the dataset
        config: Dataset configuration
        num_questions: Number of questions to sample
        seed: Random seed for reproducibility

    Returns:
        Tuple of (corpus_list, questions_list)
    """
    print(f"\n{'=' * 60}")
    print(f"Processing {dataset_name.upper()}")
    print(f"{'=' * 60}")

    # Load full dataset
    print(f"Loading {config['hf_name']}...")
    corpus_data = load_dataset(config["hf_name"], "corpus", split="train")
    questions_data = load_dataset(config["hf_name"], "questions", split="train")

    print(f"Full dataset: {len(corpus_data)} documents, {len(questions_data)} questions")

    # Build corpus text lookup
    corpus_texts = [get_text_from_doc(doc, config["corpus_text_field"]) for doc in corpus_data]

    # Map each question to its document(s)
    # doc_to_questions: doc_idx -> list of question indices
    doc_to_questions = defaultdict(list)
    question_to_doc = {}  # question_idx -> doc_idx (first found)

    print("Mapping questions to documents...")
    for q_idx, question in enumerate(questions_data):
        passage = get_passage_from_question(question, config["passage_field"])
        if not passage:
            continue

        # Find which document contains this passage
        for doc_idx, doc_text in enumerate(corpus_texts):
            if passage in doc_text:
                doc_to_questions[doc_idx].append(q_idx)
                if q_idx not in question_to_doc:
                    question_to_doc[q_idx] = doc_idx
                break  # Only map to first document found

    valid_questions = len(question_to_doc)
    print(f"Valid questions (passage exists in corpus): {valid_questions}/{len(questions_data)}")

    # Rank documents by question count (descending)
    docs_by_question_count = sorted(
        doc_to_questions.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )

    print(f"Top 5 documents by question count:")
    for doc_idx, q_list in docs_by_question_count[:5]:
        print(f"  Doc {doc_idx}: {len(q_list)} questions")

    # Select fewest documents to get ≥num_questions
    selected_doc_indices = []
    selected_question_indices = []

    for doc_idx, q_list in docs_by_question_count:
        selected_doc_indices.append(doc_idx)
        selected_question_indices.extend(q_list)

        if len(selected_question_indices) >= num_questions:
            break

    print(f"Selected {len(selected_doc_indices)} documents with {len(selected_question_indices)} questions")

    # Sample exactly num_questions from the selected questions
    rng = random.Random(seed)
    n = min(num_questions, len(selected_question_indices))

    if len(selected_question_indices) > num_questions:
        sampled_q_indices = rng.sample(selected_question_indices, n)
    else:
        sampled_q_indices = selected_question_indices

    # Get the actual question and document data
    sampled_questions = [dict(questions_data[i]) for i in sampled_q_indices]

    # Only keep documents that contain the sampled questions' passages
    sampled_passages = [
        get_passage_from_question(q, config["passage_field"])
        for q in sampled_questions
    ]

    # Find minimal set of documents for sampled questions
    needed_doc_indices = set()
    for passage in sampled_passages:
        for doc_idx in selected_doc_indices:
            if passage in corpus_texts[doc_idx]:
                needed_doc_indices.add(doc_idx)
                break

    needed_docs = [dict(corpus_data[i]) for i in sorted(needed_doc_indices)]

    print(f"Final: {len(needed_docs)} documents, {len(sampled_questions)} questions")

    # Verify coverage
    covered = 0
    for passage in sampled_passages:
        for doc in needed_docs:
            doc_text = get_text_from_doc(doc, config["corpus_text_field"])
            if passage and passage in doc_text:
                covered += 1
                break

    print(f"Coverage: {covered}/{len(sampled_passages)} passages found in corpus")

    return needed_docs, [dict(q) for q in sampled_questions]


def main():
    parser = argparse.ArgumentParser(description="Generate nano MTCB datasets")
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push to HuggingFace Hub (default: dry run)",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=100,
        help="Number of questions per dataset (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS.keys()),
        help="Datasets to process (default: all)",
    )
    args = parser.parse_args()

    print(f"Generating nano datasets with {args.num_questions} questions each")
    print(f"Random seed: {args.seed}")
    print(f"Push to HuggingFace: {args.push}")
    print(f"Datasets: {args.datasets}")

    summary = []

    for dataset_name in args.datasets:
        if dataset_name not in DATASETS:
            print(f"Unknown dataset: {dataset_name}, skipping")
            continue

        config = DATASETS[dataset_name]

        try:
            corpus, questions = generate_nano_dataset(
                dataset_name,
                config,
                num_questions=args.num_questions,
                seed=args.seed,
            )

            summary.append({
                "dataset": dataset_name,
                "corpus_size": len(corpus),
                "questions_size": len(questions),
            })

            if args.push:
                # Create HuggingFace datasets
                corpus_dataset = Dataset.from_list(corpus)
                questions_dataset = Dataset.from_list(questions)

                # Create DatasetDict with configs
                nano_name = f"chonkie-ai/nano-{dataset_name}"

                print(f"\nPushing to {nano_name}...")

                # Push corpus config
                corpus_dict = DatasetDict({"train": corpus_dataset})
                corpus_dict.push_to_hub(nano_name, config_name="corpus")

                # Push questions config
                questions_dict = DatasetDict({"train": questions_dataset})
                questions_dict.push_to_hub(nano_name, config_name="questions")

                print(f"Successfully pushed {nano_name}")

        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()

    # Print summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Dataset':<15} {'Corpus':<10} {'Questions':<10}")
    print("-" * 35)

    total_corpus = 0
    total_questions = 0
    for item in summary:
        print(f"{item['dataset']:<15} {item['corpus_size']:<10} {item['questions_size']:<10}")
        total_corpus += item["corpus_size"]
        total_questions += item["questions_size"]

    print("-" * 35)
    print(f"{'TOTAL':<15} {total_corpus:<10} {total_questions:<10}")


if __name__ == "__main__":
    main()
