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
    dataset = dataset.map(clean_dataset_text, batched=True, num_proc=4)
    dataset = dataset.map(process_categories, batched=True, num_proc=4)

    print(f"\n3. Saving base Dataset to {BASE_DATA_DIR}...")

    # Save as memory-mapped Arrow files
    dataset.save_to_disk(BASE_DATA_DIR / "arxiv_base_dataset")
    print("\nBase dataset creation completed succesfully.")


if __name__ == "__main__":
    main()