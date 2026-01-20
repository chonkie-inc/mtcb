#!/usr/bin/env python3
"""
Fix data quality issues in MTCB datasets.

Issues found:
1. TACHA (809/2065 = 39% broken):
   - Passages are truncated tables missing intermediate rows
   - The `chunk-must-contain` was extracted incorrectly during QA generation
   - Fix: Search for passages that contain ALL the unique data rows

2. GENMAICHA (156/2193 = 7% broken):
   - Passages have transcription errors vs corpus
   - Example: "a but not b" vs "a but not c"
   - Fix: Use fuzzy matching to find the closest corpus passage

3. GACHA (6/2884 = 0.2% broken):
   - 6 questions have empty `chunk-must-contain` field
   - Fix: Remove these questions or regenerate the passage

Usage:
    python scripts/fix_dataset_quality.py --dataset tacha --analyze
    python scripts/fix_dataset_quality.py --dataset tacha --fix --push
"""

import argparse
from collections import defaultdict
from datasets import Dataset, DatasetDict, load_dataset


def analyze_tacha():
    """Analyze tacha issues in detail."""
    print("=" * 60)
    print("TACHA ANALYSIS")
    print("=" * 60)

    corpus = load_dataset("chonkie-ai/tacha", "corpus", split="train")
    questions = load_dataset("chonkie-ai/tacha", "questions", split="train")
    corpus_texts = [doc["text"] for doc in corpus]

    valid = 0
    fixable = 0
    unfixable = 0

    for idx, q in enumerate(questions):
        passage = q["chunk-must-contain"]

        # Check if valid
        if any(passage in doc for doc in corpus_texts):
            valid += 1
            continue

        # Try to find the passage content in a relaxed way
        # Extract unique data (numbers, specific text) from passage
        lines = passage.split("\n")
        data_lines = [l for l in lines if l.startswith("|") and not l.startswith("|---")]

        # Check if all data lines exist in some corpus doc
        found_doc = None
        for doc in corpus_texts:
            if all(line.strip() in doc for line in data_lines if line.strip()):
                found_doc = doc
                break

        if found_doc:
            fixable += 1
        else:
            unfixable += 1
            if unfixable <= 5:
                print(f"\nUnfixable Q{idx}:")
                print(f"  Passage: {passage[:150]}...")

    print(f"\nResults:")
    print(f"  Valid: {valid}")
    print(f"  Fixable (data lines found): {fixable}")
    print(f"  Unfixable: {unfixable}")
    print(f"  Total: {len(questions)}")


def fix_tacha(push=False):
    """Fix tacha by reconstructing correct passages from corpus."""
    print("=" * 60)
    print("FIXING TACHA")
    print("=" * 60)

    corpus = load_dataset("chonkie-ai/tacha", "corpus", split="train")
    questions = load_dataset("chonkie-ai/tacha", "questions", split="train")
    corpus_texts = [doc["text"] for doc in corpus]

    fixed_questions = []
    already_valid = 0
    reconstructed = 0
    removed = 0

    for idx, q in enumerate(questions):
        passage = q["chunk-must-contain"]
        q_dict = dict(q)

        # Check if already valid
        if any(passage in doc for doc in corpus_texts):
            fixed_questions.append(q_dict)
            already_valid += 1
            continue

        # Extract data lines from passage
        lines = passage.split("\n")
        data_lines = [l.strip() for l in lines if l.startswith("|") and not l.startswith("|---") and l.strip()]

        if not data_lines:
            removed += 1
            continue

        # Find corpus doc containing all data lines
        found = False
        for doc in corpus_texts:
            if all(line in doc for line in data_lines):
                # Find the smallest substring containing all data lines
                first_pos = min(doc.find(line) for line in data_lines)
                last_pos = max(doc.find(line) + len(line) for line in data_lines)

                # Extend backwards to find table start
                search_start = max(0, first_pos - 500)

                # Look for table markers
                actual_start = first_pos
                for marker in ["## Table", "|  |", "| "]:
                    pos = doc.rfind(marker, search_start, first_pos)
                    if pos >= 0:
                        # Find start of this line
                        line_start = doc.rfind("\n", search_start, pos)
                        if line_start >= 0:
                            actual_start = min(actual_start, line_start + 1)
                        else:
                            actual_start = min(actual_start, pos)

                new_passage = doc[actual_start:last_pos]

                # Verify reconstruction contains all data
                if new_passage and all(line in new_passage for line in data_lines):
                    q_dict["chunk-must-contain"] = new_passage
                    fixed_questions.append(q_dict)
                    reconstructed += 1
                    found = True
                    break

        if not found:
            removed += 1

    print(f"Already valid: {already_valid}")
    print(f"Reconstructed: {reconstructed}")
    print(f"Removed: {removed}")
    print(f"Total fixed: {len(fixed_questions)}")

    if push:
        print("\nPushing fixed dataset...")
        questions_ds = Dataset.from_list(fixed_questions)
        ds_dict = DatasetDict({"train": questions_ds})
        ds_dict.push_to_hub("chonkie-ai/tacha", config_name="questions")
        print("Done!")

    return fixed_questions


def analyze_genmaicha():
    """Analyze genmaicha issues."""
    print("=" * 60)
    print("GENMAICHA ANALYSIS")
    print("=" * 60)

    corpus = load_dataset("chonkie-ai/genmaicha", "corpus", split="train")
    questions = load_dataset("chonkie-ai/genmaicha", "questions", split="train")
    corpus_texts = [doc["text"] for doc in corpus]

    valid = 0
    missing = []

    for idx, q in enumerate(questions):
        passage = q["chunk-must-contain"]

        if any(passage in doc for doc in corpus_texts):
            valid += 1
        else:
            missing.append((idx, passage))

    print(f"Valid: {valid}")
    print(f"Missing: {len(missing)}")
    print(f"\nSample missing passages:")

    for idx, passage in missing[:5]:
        print(f"\nQ{idx}: {passage[:100]}...")
        # Try to find similar
        words = passage.split()[:6]
        search = " ".join(words)
        for doc in corpus_texts:
            if search in doc:
                loc = doc.find(search)
                print(f"  Similar in corpus: {doc[loc:loc+100]}...")
                break


def fix_genmaicha(push=False):
    """Fix genmaicha by removing questions with bad passages."""
    print("=" * 60)
    print("FIXING GENMAICHA")
    print("=" * 60)

    corpus = load_dataset("chonkie-ai/genmaicha", "corpus", split="train")
    questions = load_dataset("chonkie-ai/genmaicha", "questions", split="train")
    corpus_texts = [doc["text"] for doc in corpus]

    fixed_questions = []
    removed = 0

    for q in questions:
        passage = q["chunk-must-contain"]
        if passage and any(passage in doc for doc in corpus_texts):
            fixed_questions.append(dict(q))
        else:
            removed += 1

    print(f"Kept: {len(fixed_questions)}")
    print(f"Removed: {removed}")

    if push:
        print("\nPushing fixed dataset...")
        questions_ds = Dataset.from_list(fixed_questions)
        ds_dict = DatasetDict({"train": questions_ds})
        ds_dict.push_to_hub("chonkie-ai/genmaicha", config_name="questions")
        print("Done!")

    return fixed_questions


def fix_gacha(push=False):
    """Fix gacha by removing questions with empty passages."""
    print("=" * 60)
    print("FIXING GACHA")
    print("=" * 60)

    corpus = load_dataset("chonkie-ai/gacha", "corpus", split="train")
    questions = load_dataset("chonkie-ai/gacha", "questions", split="train")
    corpus_texts = [doc["text"] for doc in corpus]

    fixed_questions = []
    removed = 0

    for q in questions:
        passage = q["chunk-must-contain"]
        if passage and any(passage in doc for doc in corpus_texts):
            fixed_questions.append(dict(q))
        else:
            removed += 1

    print(f"Kept: {len(fixed_questions)}")
    print(f"Removed: {removed}")

    if push:
        print("\nPushing fixed dataset...")
        questions_ds = Dataset.from_list(fixed_questions)
        ds_dict = DatasetDict({"train": questions_ds})
        ds_dict.push_to_hub("chonkie-ai/gacha", config_name="questions")
        print("Done!")

    return fixed_questions


def main():
    parser = argparse.ArgumentParser(description="Fix MTCB dataset quality issues")
    parser.add_argument("--dataset", choices=["tacha", "genmaicha", "gacha", "all"], required=True)
    parser.add_argument("--analyze", action="store_true", help="Analyze issues only")
    parser.add_argument("--fix", action="store_true", help="Fix issues")
    parser.add_argument("--push", action="store_true", help="Push fixed dataset to HuggingFace")
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset != "all" else ["tacha", "genmaicha", "gacha"]

    for ds in datasets:
        if args.analyze:
            if ds == "tacha":
                analyze_tacha()
            elif ds == "genmaicha":
                analyze_genmaicha()
            elif ds == "gacha":
                print("GACHA: 6 questions have empty chunk-must-contain")

        if args.fix:
            if ds == "tacha":
                fix_tacha(push=args.push)
            elif ds == "genmaicha":
                fix_genmaicha(push=args.push)
            elif ds == "gacha":
                fix_gacha(push=args.push)


if __name__ == "__main__":
    main()
