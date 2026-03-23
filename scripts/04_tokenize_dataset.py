"""
Tokenize the taxonomy dataset and save a PyTorch-ready Hugging Face dataset.

This script:
1. Resolves tokenizer name from a YAML config file (required).
2. Loads the taxonomy train/val/test dataset from disk (path required via CLI).
3. Tokenizes `title` + `abstract` as paired inputs.
4. Removes raw text/metadata columns after tokenization.
5. Sets PyTorch tensor formatting for training and evaluation.
6. Saves the tokenized dataset to a model-specific output directory.
7. Writes tokenization metadata for reproducibility.

Required arguments:
- `--config`: Path to YAML config file (must contain `model.pretrained_name`)
- `--dataset-path`: Path to input dataset directory (HuggingFace format)
- `--output-dir`: Path to save tokenized dataset

Output:
- Tokenized dataset at the specified output directory
- `tokenization_meta.json` inside output directory

Example usage:
    python scripts/04_tokenize_dataset.py \
        --config configs/scibert_classification.yaml \
        --dataset-path data/processed/arxiv_taxonomy_dataset \
        --output-dir data/processed/tok_scibert_scivocab_uncased
"""


import os
import json
import yaml
import argparse
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from transformers.utils import logging
logging.set_verbosity_error()

from datasets import load_from_disk
from transformers import AutoTokenizer

from arxiv_paper_discovery.data import tokenize_batch
from arxiv_paper_discovery.config import PROCESSED_DATA_DIR

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to the input dataset directory (HuggingFace format)")
    parser.add_argument("--output-dir", type=str, required=True)
    return parser.parse_args()

# Helper functions removed: tokenizer name is read directly from the provided config

def main():
    args = parse_args()

    # Read tokenizer name from provided config (no CLI override allowed)
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    try:
        tokenizer_name = config["model"]["pretrained_name"]
    except Exception as exc:
        raise ValueError(
            f"Could not read 'model.pretrained_name' from config: {args.config}"
        ) from exc


    dataset_path = Path(args.dataset_path)

    print(f"\n1. Loading category mapped dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)

    print(f"\n2. Tokenizing with {tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    tokenized_dataset = dataset.map(
        tokenize_batch,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=6,
        desc="Tokenizing text"
    )

    print("\n3. Formatting for PyTorch...")
    # Drop raw text/metadata to save space and prevent DataLoader collation errors
    columns_to_remove = ["id", "title", "abstract", "categories", "authors", "update_date"]
    
    # Verify columns exist before removing them to avoid errors
    existing_cols = tokenized_dataset["train"].column_names
    cols_to_drop = [c for c in columns_to_remove if c in existing_cols]
    
    tokenized_dataset = tokenized_dataset.remove_columns(cols_to_drop)
    
    tokenized_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "token_type_ids", "labels"]
    )

    output_dir = Path(args.output_dir)
    print(f"\n4. Adding tokenizer metadata to dataset info and saving to {output_dir}...")
    
    tokenized_dataset.save_to_disk(output_dir)

    metadata = {
        "tokenizer_name": tokenizer_name,
        "config_path": args.config,
        "dataset_path": str(dataset_path),
        "output_dir": str(output_dir),
    }
    with open(output_dir / "tokenization_meta.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nTokenization complete.")

if __name__ == "__main__":
    main()
