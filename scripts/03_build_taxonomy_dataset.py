"""
Build the taxonomy-labeled train/validation/test dataset from the base ArXiv dataset.

This script:
1. Loads the base dataset from disk.
2. Maps raw ArXiv category tags to human-readable taxonomy group name lists.
3. Filters out papers that do not map to any taxonomy group.
4. Splits data into train/val/test (80/10/10).
5. Saves the split dataset for downstream tokenization.

Note: multi-hot encoding is deliberately deferred to 04_tokenize_dataset.py so
that the taxonomy dataset remains human-readable and inspectable.

Input:
- `data/base/arxiv_base_dataset`

Output:
- `data/processed/arxiv_taxonomy_dataset`  (labels column: list of group name strings)

Run:
- `python scripts/03_build_taxonomy_dataset.py --seed 42`
"""

import os
import argparse

NUM_PROC = min(6, os.cpu_count() or 1)


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from datasets import load_from_disk, DatasetDict

from arxiv_paper_discovery.label_taxonomy import categories_to_labels
from arxiv_paper_discovery.config import BASE_DATA_PATH, PROCESSED_DATA_DIR


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def apply_taxonomy(batch):
    """Maps raw ArXiv categories to taxonomy group name lists."""
    batch_labels = [categories_to_labels(cats) for cats in batch["categories"]]
    return {"labels": batch_labels}

def has_valid_label(example):
    """Filters out papers with no matching taxonomy groups."""
    return len(example["labels"]) > 0

def main():
    args = parse_args()
    seed = args.seed

    print(f"Using random seed: {seed}")

    print("1. Loading the Base Dataset...")
    dataset = load_from_disk(BASE_DATA_PATH)

    print("\n2. Applying Label Taxonomy (Group Name Mapping)...")
    dataset = dataset.map(
        apply_taxonomy,
        batched=True,
        num_proc=NUM_PROC,
        desc="Mapping categories to taxonomy (labels)"
    )

    print("\n3. Filtering dataset to remove out-of-taxonomy papers...")
    original_size = len(dataset)
    dataset = dataset.filter(
        has_valid_label,
        num_proc=NUM_PROC,
        desc="Filtering empty labels"
    )
    print(f"   Retained {len(dataset)} / {original_size} papers.")


    print("\n4. Splitting into Train/Val/Test (80/10/10)...")
    train_test = dataset.train_test_split(test_size=0.2, seed=seed, shuffle=True)
    test_val = train_test['test'].train_test_split(test_size=0.5, seed=seed, shuffle=True)

    final_dataset = DatasetDict({
        'train': train_test['train'],
        'val': test_val['train'],
        'test': test_val['test']
    })

    print(f"\n5. Saving category mapped dataset to {PROCESSED_DATA_DIR}...")
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save the core dataset (keeps text columns intact for future flexibility)
    dataset_path = PROCESSED_DATA_DIR / "arxiv_taxonomy_dataset"
    final_dataset.save_to_disk(dataset_path)

    print(f"\nSuccess! Taxonomy dataset saved to {dataset_path}")

if __name__ == "__main__":
    main()
