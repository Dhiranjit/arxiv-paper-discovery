"""
Sample a random train/val subset from a tokenized SciBERT dataset.

Defaults:
- Input dataset: data/processed/tok_scibert_scivocab_uncased
- Output dataset: upload/tok_scibert_train_val_200k
- Total samples: 200000

Examples:
python3 scripts/sample_tok_scibert_train_val.py

python3 scripts/sample_dataset.py \
--num-samples 300000 \
--input-dir data/processed/tok_scibert_scivocab_uncased \
--output-dir upload/tok_scibert_train_val_300k
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import DatasetDict, load_from_disk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample train/val from tokenized SciBERT data.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/processed/tok_scibert_scivocab_uncased"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("upload/tok_scibert_train_val_200k"),
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=200_000,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    num_samples = args.num_samples

    if num_samples <= 0:
        raise ValueError("--num-samples must be > 0.")
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    dataset = load_from_disk(str(input_dir))
    if "train" not in dataset or "val" not in dataset:
        raise ValueError(
            f"Expected dataset with 'train' and 'val' splits. Found: {list(dataset.keys())}"
        )

    train_ds = dataset["train"]
    val_ds = dataset["val"]

    train_len = len(train_ds)
    val_len = len(val_ds)
    total = train_len + val_len
    target_total = min(num_samples, total)

    train_target = min(int(target_total * train_len / total), train_len)
    val_target = min(target_total - train_target, val_len)

    sampled_train = train_ds.shuffle(seed=42).select(range(train_target))
    sampled_val = val_ds.shuffle(seed=43).select(range(val_target))

    sampled_dataset = DatasetDict({"train": sampled_train, "val": sampled_val})

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    sampled_dataset.save_to_disk(str(args.output_dir))

    print(f"Input:  {input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Sampled train: {train_target} / {train_len}")
    print(f"Sampled val:   {val_target} / {val_len}")
    print(f"Total:         {train_target + val_target}")


if __name__ == "__main__":
    main()
