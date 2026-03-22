
"""
Build the base ArXiv dataset used by downstream preprocessing and training.

This script:
1. Loads the raw Kaggle ArXiv JSON snapshot.
2. Keeps only the columns needed by the project pipeline.
3. Cleans `title` and `abstract` text (URL removal + whitespace normalization).
4. Converts the space-delimited `categories` string into a list of categories.
5. Saves the result as a memory-mapped Hugging Face Dataset on disk.

Input:
- `data/raw/arxiv/arxiv-metadata-oai-snapshot.json`

Output:
- `data/base/arxiv_base_dataset`

Run:
- `python scripts/02_create_base_dataset.py`
"""

import os
from pathlib import Path
from datasets import load_dataset

from arxiv_paper_discovery.data import clean_dataset_text, process_categories
from arxiv_paper_discovery.config import RAW_DATA_PATH, BASE_DATA_DIR


def main():
    print("\n1. Loading raw JSON dataset...")
    # Using Hugging Face Datasets to load the large JSON file efficiently
    dataset = load_dataset("json", data_files=str(RAW_DATA_PATH), split="train")

    # Select only the relevant columns to reduce memory usage
    cols_to_keep = ['id', 'title', 'abstract', 'categories', 'authors', 'update_date']
    dataset = dataset.select_columns(cols_to_keep)

    print("\n2. Cleaning base text and formatting categories...")
    dataset = dataset.map(clean_dataset_text, batched=True, num_proc=6)
    dataset = dataset.map(process_categories, batched=True, num_proc=6)

    print(f"\n3. Saving base Dataset to {BASE_DATA_DIR}...")

    # Save as memory-mapped Arrow files
    dataset.save_to_disk(BASE_DATA_DIR / "arxiv_base_dataset")
    print("\nBase dataset creation completed succesfully.")


if __name__ == "__main__":
    main()
