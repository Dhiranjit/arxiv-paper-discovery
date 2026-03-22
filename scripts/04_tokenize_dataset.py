"""
Tokenize the taxonomy dataset and save a PyTorch-ready Hugging Face dataset.

This script:
1. Resolves tokenizer name from CLI or a YAML config file.
2. Loads the taxonomy train/val/test dataset from disk.
3. Tokenizes `title` + `abstract` as paired inputs.
4. Removes raw text/metadata columns after tokenization.
5. Sets PyTorch tensor formatting for training and evaluation.
6. Saves the tokenized dataset to a model-specific output directory.
7. Writes tokenization metadata for reproducibility.

Input:
- `data/processed/arxiv_taxonomy_dataset`
- Config file passed via `--config` (optional, for `model.pretrained_name`)
- Tokenizer name passed via `--tokenizer-name` (optional override)

Output:
- `data/processed/tok_<model_name>`
- `tokenization_meta.json` inside output directory

Run:
- `python scripts/04_tokenize_dataset.py --config configs/scibert_classification.yaml`
- `python scripts/04_tokenize_dataset.py --tokenizer-name allenai/scibert_scivocab_uncased`

- output_dir can also be specified to override default naming:
"""


import os
import re
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
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--tokenizer-name", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()

def model_name_to_slug(name: str) -> str:
    """Convert model/tokenizer id to a filesystem-safe directory suffix."""
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", name.strip())

def resolve_tokenizer_name(args) -> str:
    """Resolve tokenizer name from CLI first, then fallback to config file."""
    if args.tokenizer_name:
        return args.tokenizer_name

    if args.config is None:
        raise ValueError("Provide either --tokenizer-name or --config.")

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    try:
        return config["model"]["pretrained_name"]
    except Exception as exc:
        raise ValueError(
            f"Could not read 'model.pretrained_name' from config: {args.config}"
        ) from exc

def main():
    args = parse_args()
    tokenizer_name = resolve_tokenizer_name(args)

    dataset_path = PROCESSED_DATA_DIR / "arxiv_taxonomy_dataset"

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

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        model_safe_name = model_name_to_slug(tokenizer_name)
        output_dir = PROCESSED_DATA_DIR / f"tok_{model_safe_name}"
    
    print(f"\n4. Saving PyTorch-ready dataset to {output_dir}...")
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
