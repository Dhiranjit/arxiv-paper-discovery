"""
Tokenize the taxonomy dataset and save a PyTorch-ready Hugging Face dataset.

This script:
1. Resolves tokenizer name from a YAML config file (required).
2. Loads the taxonomy train/val/test dataset from disk (path required via CLI).
3. Encodes string group-name labels as multi-hot vectors.
4. Tokenizes `title` + `abstract` as paired inputs.
5. Removes raw text/metadata columns after tokenization.
6. Saves the tokenized dataset to a model-specific output directory.
8. Writes tokenization metadata and group_to_index.json for reproducibility.

Required arguments:
- `--config`: Path to YAML config file (must contain `model.pretrained_name`)
- `--dataset-path`: Path to input dataset directory (HuggingFace format)
- `--output-dir`: Path to save tokenized dataset

Outputs:
- Tokenized dataset at the specified output directory
- `tokenization_meta.json` inside output directory
- `group_to_index.json` inside output directory

Example usage:
python scripts/04_tokenize_dataset.py \
--config configs/scibert.yaml \
--dataset-path data/processed/arxiv_taxonomy_dataset \
--output-dir data/processed/tok_scibert_scivocab_uncased
"""


import os
import json
import yaml
import argparse
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
NUM_PROC = min(6, os.cpu_count() or 1)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from transformers.utils import logging
logging.set_verbosity_error()

from datasets import load_from_disk
from transformers import AutoTokenizer

from arxiv_paper_discovery.data import tokenize_batch
from arxiv_paper_discovery.label_taxonomy import labels_to_multihot, LABEL_TO_IDX


def encode_labels(batch):
    """Convert group name lists to multi-hot vectors."""
    return {"labels": [labels_to_multihot(groups) for groups in batch["labels"]]}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to the input dataset directory (HuggingFace format)")
    parser.add_argument("--output-dir", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    # Read tokenizer name from provided config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    try:
        tokenizer_name = config["model"]["pretrained_name"]
    except KeyError as exc:
        raise ValueError(
            f"Could not read 'model.pretrained_name' from config: {args.config}"
        ) from exc


    dataset_path = Path(args.dataset_path)

    print(f"\n1. Loading category mapped dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)

    print("\n2. Encoding labels as multi-hot vectors...")
    dataset = dataset.map(encode_labels, batched=True, num_proc=NUM_PROC, desc="Encoding labels")

    print(f"\n3. Tokenizing with {tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    tokenized_dataset = dataset.map(
        tokenize_batch,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=NUM_PROC,
        desc="Tokenizing text"
    )

    print("\n4. Dropping raw text/metadata columns...")
    # Drop raw text/metadata to save space and prevent DataLoader collation errors
    columns_to_remove = ["id", "title", "abstract", "categories", "authors", "update_date"]
    existing_cols = tokenized_dataset["train"].column_names
    
    cols_to_drop = [c for c in columns_to_remove if c in existing_cols]
    tokenized_dataset = tokenized_dataset.remove_columns(cols_to_drop)

    output_dir = Path(args.output_dir)
    print(f"\n5. Saving tokenized dataset and metadata to {output_dir}...")

    tokenized_dataset.save_to_disk(output_dir)

    metadata = {
        "tokenizer_name": tokenizer_name,
        "config_path": args.config,
        "dataset_path": str(dataset_path),
        "output_dir": str(output_dir),
    }
    with open(output_dir / "tokenization_meta.json", "w") as f:
        json.dump(metadata, f, indent=2)

    with open(output_dir / "group_to_index.json", "w") as f:
        json.dump(LABEL_TO_IDX, f)

    print("\nTokenization complete.")

if __name__ == "__main__":
    main()
