#!/usr/bin/env python3
"""Fix the hojicha dataset on HuggingFace by re-uploading with proper configs.

The hojicha dataset has corpus and questions parquet files but they're not
properly configured as separate dataset configs.

Usage:
    python scripts/fix_hojicha_dataset.py --push  # Fix and push to HuggingFace
    python scripts/fix_hojicha_dataset.py         # Dry run
"""

import argparse

import pandas as pd
from datasets import Dataset, DatasetDict


def main():
    parser = argparse.ArgumentParser(description="Fix hojicha dataset configs")
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push to HuggingFace Hub (default: dry run)",
    )
    args = parser.parse_args()

    # Read parquet files from cache
    corpus_path = "/Users/bhavnick/.cache/huggingface/hub/datasets--chonkie-ai--hojicha/snapshots/e59835f9481f9a5f39b6493c32946e12bb0405ef/corpus/train-00000-of-00001.parquet"
    questions_path = "/Users/bhavnick/.cache/huggingface/hub/datasets--chonkie-ai--hojicha/snapshots/e59835f9481f9a5f39b6493c32946e12bb0405ef/questions/train-00000-of-00001.parquet"

    print("Loading corpus parquet...")
    corpus_df = pd.read_parquet(corpus_path)
    print(f"Corpus: {len(corpus_df)} documents")
    print(f"Columns: {corpus_df.columns.tolist()}")

    print("\nLoading questions parquet...")
    questions_df = pd.read_parquet(questions_path)
    print(f"Questions: {len(questions_df)} questions")
    print(f"Columns: {questions_df.columns.tolist()}")

    if args.push:
        # Create HuggingFace datasets
        corpus_dataset = Dataset.from_pandas(corpus_df)
        questions_dataset = Dataset.from_pandas(questions_df)

        print("\nPushing corpus config...")
        corpus_dict = DatasetDict({"train": corpus_dataset})
        corpus_dict.push_to_hub("chonkie-ai/hojicha", config_name="corpus")

        print("Pushing questions config...")
        questions_dict = DatasetDict({"train": questions_dataset})
        questions_dict.push_to_hub("chonkie-ai/hojicha", config_name="questions")

        print("\nDone! Hojicha dataset now has corpus and questions configs.")
    else:
        print("\nDry run - use --push to upload to HuggingFace")


if __name__ == "__main__":
    main()
