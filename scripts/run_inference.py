"""
scripts/batch_infer.py

Run batch inference over an untokenized dataset and write predictions to JSONL.

Input  : HuggingFace dataset split saved to disk  OR  a raw JSONL file.
         Each record must have a "title" field; "abstract" is optional.

Output : JSONL where every line is the original record extended with:
             "predicted_tags"   : list[str]
             "tag_probabilities": dict[str, float]
         and, when present/decodable from input labels:
             "true_tags"        : list[str]

Design notes
------------
- Processes data in batches without loading the entire input into memory.
- Supports both dataset-first and file-first workflows:
  - `--input-hf` for HuggingFace datasets saved via save_to_disk()
  - `--input-jsonl` for raw line-delimited JSON
- `--limit` is intended for quick smoke tests over the first N records.

Usage examples
--------------
# From a HuggingFace dataset split saved to disk:
python scripts/batch_infer.py \
    --checkpoint saved_models/my_run \
    --input-hf   data/processed/arxiv_taxonomy_dataset \
    --split      test \
    --output     outputs/test_predictions.jsonl \
    --batch-size 64 \
    --limit      500 \
    --threshold 0.35

# From a raw JSONL file:
python scripts/batch_infer.py \
    --checkpoint saved_models/my_run \
    --input-jsonl data/raw/papers.jsonl \
    --output      outputs/papers_tagged.jsonl \
    --limit       200
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
from transformers.utils import logging as hf_logging
warnings.filterwarnings("ignore", category=FutureWarning)
hf_logging.set_verbosity_error()

import argparse
import json
import time
from collections.abc import Iterator
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk
from tqdm import tqdm

from arxiv_paper_discovery.predictor import ArticleTagger

GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_multihot(multihot: list[int | float], index_to_class: dict[int, str]) -> list[str]:
    """Convert a multi-hot binary vector to a list of class name strings."""
    return [index_to_class[i] for i, val in enumerate(multihot) if float(val) > 0.5]


def _load_hf_split(dataset_dir: Path, split: str) -> Dataset:
    """Load a HuggingFace dataset split."""
    ds = load_from_disk(str(dataset_dir))
    if isinstance(ds, DatasetDict):
        if split not in ds:
            available_splits = ", ".join(ds.keys())
            raise ValueError(
                f"Split '{split}' not found in {dataset_dir}. "
                f"Available splits: {available_splits}."
            )
        return ds[split]
    return ds


def _iter_hf_batches(
    split_ds: Dataset,
    batch_size: int,
    limit: int | None,
) -> Iterator[list[dict]]:
    """Yield fixed-size batches from a HuggingFace dataset split."""
    max_records = min(len(split_ds), limit) if limit is not None else len(split_ds)
    for start in range(0, max_records, batch_size):
        end = min(start + batch_size, max_records)
        yield split_ds.select(range(start, end)).to_list()


def _iter_jsonl_batches(
    file_path: Path,
    batch_size: int,
    limit: int | None,
) -> Iterator[list[dict]]:
    """Yield fixed-size batches from a JSONL file without loading the full file."""
    emitted = 0
    batch: list[dict] = []

    with open(file_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            batch.append(json.loads(line))
            if len(batch) < batch_size:
                continue

            if limit is not None and emitted + len(batch) > limit:
                batch = batch[: limit - emitted]
            if batch:
                yield batch
                emitted += len(batch)
            if limit is not None and emitted >= limit:
                return
            batch = []

    if batch and (limit is None or emitted < limit):
        if limit is not None:
            batch = batch[: limit - emitted]
        if batch:
            yield batch


def _extract_true_tags(record: dict, index_to_class: dict[int, str]) -> list[str] | None:
    """Return decoded true tags when labels are present and usable; otherwise None."""
    labels = record.get("labels")
    if labels is None or not isinstance(labels, list):
        return None
    if not all(isinstance(x, (int, float)) for x in labels):
        return None
    if len(labels) != len(index_to_class):
        return None
    return _decode_multihot(labels, index_to_class)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch inference over raw arxiv papers."
    )

    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Path to a Hugging Face model directory saved via save_pretrained().",
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input-hf", type=Path, metavar="HF_DATASET_DIR",
        help="Root directory of a HuggingFace dataset saved with save_to_disk().",
    )
    input_group.add_argument(
        "--input-jsonl", type=Path, metavar="JSONL_FILE",
        help="Path to a raw JSONL file (one JSON object per line, must have 'title').",
    )

    parser.add_argument(
        "--split", type=str, default="test",
        help="Which dataset split to use when --input-hf is set (default: test).",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Destination JSONL file for predictions.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Articles per forward pass (default: 32).",
    )
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Maximum number of records to process (default: no limit).",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Sigmoid prediction threshold (default: 0.5).",
    )

    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be a positive integer.")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be a positive integer.")

    # ---- Build batch iterator ----
    if args.input_hf:
        if not args.input_hf.exists():
            raise FileNotFoundError(f"HF dataset not found: {args.input_hf}")
        split_ds = _load_hf_split(args.input_hf, args.split)
        source_label = f"HF dataset ({args.input_hf.name}/{args.split})"
        target_total = min(len(split_ds), args.limit) if args.limit is not None else len(split_ds)
        batches = _iter_hf_batches(split_ds, args.batch_size, args.limit)
        total_batches = (target_total + args.batch_size - 1) // args.batch_size if target_total else 0
    else:
        if not args.input_jsonl.exists():
            raise FileNotFoundError(f"JSONL file not found: {args.input_jsonl}")
        source_label = str(args.input_jsonl)
        target_total = args.limit
        batches = _iter_jsonl_batches(args.input_jsonl, args.batch_size, args.limit)
        total_batches = None

    # ---- Load model ----
    tagger = ArticleTagger(args.checkpoint, threshold=args.threshold)

    # ---- Banner ----
    print(f"\n{CYAN}{'='*60}{RESET}")
    print(f"{CYAN}Batch inference{RESET}")
    print(f"  Source      : {source_label}")
    print(f"  Output      : {args.output}")
    if target_total is None:
        print("  Articles    : unknown (streaming JSONL)")
    else:
        print(f"  Articles    : {target_total}")
    print(f"  Limit       : {args.limit if args.limit is not None else 'none'}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  Threshold   : {tagger.threshold}")
    print(f"{CYAN}{'='*60}{RESET}\n")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    processed = 0

    with open(args.output, "w", encoding="utf-8") as out_f:
        progress_total = {"total": total_batches} if total_batches is not None else {}
        for batch in tqdm(batches, desc="Batches", **progress_total):
            titles    = [item.get("title", "") for item in batch]
            abstracts = [item.get("abstract", "") for item in batch]

            predictions = tagger.predict(titles, abstracts)

            for record, pred in zip(batch, predictions):
                out_record = {
                    **record,
                    "predicted_tags":    pred["tags"],
                    "tag_probabilities": pred["probabilities"],
                }
                true_tags = _extract_true_tags(record, tagger.index_to_class)
                if true_tags is not None:
                    out_record["true_tags"] = true_tags
                out_f.write(json.dumps(out_record) + "\n")
            processed += len(batch)

    elapsed    = time.time() - start
    throughput = processed / elapsed if elapsed > 0 else 0

    print(f"\n{GREEN}Done!{RESET}")
    print(f"  Predictions saved : {args.output}")
    print(f"  Total processed   : {processed}")
    print(f"  Elapsed           : {elapsed:.1f}s  ({throughput:.1f} articles/s)\n")


if __name__ == "__main__":
    main()
