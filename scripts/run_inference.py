"""
scripts/run_inference.py

Run batch inference over an untokenized dataset and write predictions to JSONL.

Input  : HuggingFace dataset split saved to disk  OR  a raw JSONL file.
         Each record must have a "title" field; "abstract" is optional.

Output : JSONL where every line is the original record extended with:
             "predicted_tags"   : list[str]
             "tag_probabilities": dict[str, float]
         and, when present/decodable from input labels:
             "true_tags"        : list[str]

Thresholds are loaded automatically from thresholds.json in the checkpoint
directory (per-class). Falls back to scalar 0.5 if the file is absent.

Usage examples
--------------
# From a HuggingFace dataset split saved to disk:
python scripts/run_inference.py \
    --checkpoint saved_models/my_run/best_model \
    --input-hf   data/processed/arxiv_taxonomy_dataset \
    --split      test \
    --output     outputs/test_predictions.jsonl \
    --batch-size 64 \
    --limit      500

# From a raw JSONL file:
python scripts/run_inference.py \
    --checkpoint saved_models/my_run/best_model \
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

def _load_hf_split(dataset_dir: Path, split: str) -> Dataset:
    ds = load_from_disk(str(dataset_dir))
    if isinstance(ds, DatasetDict):
        return ds[split]
    return ds


def _iter_hf_batches(split_ds: Dataset, batch_size: int, limit: int | None) -> Iterator[list[dict]]:
    max_records = min(len(split_ds), limit) if limit is not None else len(split_ds)
    for start in range(0, max_records, batch_size):
        yield split_ds.select(range(start, min(start + batch_size, max_records))).to_list()


def _iter_jsonl_batches(file_path: Path, batch_size: int, limit: int | None) -> Iterator[list[dict]]:
    emitted = 0
    batch: list[dict] = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            batch.append(json.loads(line))
            if len(batch) < batch_size:
                continue
            yield batch
            emitted += len(batch)
            if limit is not None and emitted >= limit:
                return
            batch = []
    if batch:
        yield batch




# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Batch inference over raw arxiv papers.")

    parser.add_argument("--checkpoint", type=Path, required=True,
        help="Path to a Hugging Face model directory saved via save_pretrained().")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-hf", type=Path, metavar="HF_DATASET_DIR",
        help="Root directory of a HuggingFace dataset saved with save_to_disk().")
    input_group.add_argument("--input-jsonl", type=Path, metavar="JSONL_FILE",
        help="Path to a raw JSONL file (one JSON object per line, must have 'title').")

    parser.add_argument("--split", type=str, default="test",
        help="Which dataset split to use when --input-hf is set (default: test).")
    parser.add_argument("--output", type=Path, required=True,
        help="Destination JSONL file for predictions.")
    parser.add_argument("--batch-size", type=int, default=32,
        help="Articles per forward pass (default: 32).")
    parser.add_argument("--limit", type=int, default=None, metavar="N",
        help="Maximum number of records to process (default: no limit).")

    args = parser.parse_args()

    # ---- Build batch iterator ----
    if args.input_hf:
        split_ds = _load_hf_split(args.input_hf, args.split)
        split_size = len(split_ds)
        target_total = min(split_size, args.limit) if args.limit is not None else split_size
        source_label = f"HF dataset ({args.input_hf.name}/{args.split})"
        batches = _iter_hf_batches(split_ds, args.batch_size, args.limit)
        total_batches = (target_total + args.batch_size - 1) // args.batch_size
    else:
        split_size = None
        target_total = args.limit
        source_label = str(args.input_jsonl)
        batches = _iter_jsonl_batches(args.input_jsonl, args.batch_size, args.limit)
        total_batches = None

    # ---- Load model ----
    tagger = ArticleTagger(args.checkpoint)

    # ---- Banner ----
    print(f"\n{CYAN}{'='*60}{RESET}")
    print(f"{CYAN}Batch inference{RESET}")
    print(f"  Source      : {source_label}")
    print(f"  Output      : {args.output}")
    if split_size is not None:
        print(f"  Dataset size: {split_size}")
    if target_total is None:
        print("  Processing  : unknown (streaming JSONL)")
    else:
        print(f"  Processing  : {target_total}")
    print(f"  Limit       : {args.limit if args.limit is not None else 'none'}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"{CYAN}{'='*60}{RESET}\n")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    processed = 0

    all_true: list[list[str]] = []
    all_pred: list[list[str]] = []

    with open(args.output, "w", encoding="utf-8") as out_f:
        progress_total = {"total": total_batches} if total_batches is not None else {}
        for batch in tqdm(batches, desc="Batches", **progress_total):
            titles    = [item.get("title", "") for item in batch]
            abstracts = [item.get("abstract", "") for item in batch]

            predictions = tagger.predict(titles, abstracts)

            for record, pred in zip(batch, predictions):
                record_id = record.get("id") or record.get("arxiv_id") or record.get("paper_id")
                true_tags = record.get("labels")
                out_record = {
                    "id":                record_id,
                    "title":             record.get("title", ""),
                    "true_tags":         true_tags,
                    "predicted_tags":    pred["tags"],
                    "tag_probabilities": pred["probabilities"],
                }
                if true_tags is not None:
                    all_true.append(true_tags)
                    all_pred.append(pred["tags"])
                out_f.write(json.dumps(out_record) + "\n")
            processed += len(batch)

    elapsed    = time.time() - start
    throughput = processed / elapsed if elapsed > 0 else 0

    print(f"\n{GREEN}Done!{RESET}")
    print(f"  Predictions saved : {args.output}")
    print(f"  Total processed   : {processed}")
    print(f"  Elapsed           : {elapsed:.1f}s  ({throughput:.1f} articles/s)")

    if all_true:
        from sklearn.preprocessing import MultiLabelBinarizer
        from sklearn.metrics import f1_score, precision_score, recall_score
        mlb = MultiLabelBinarizer(classes=list(tagger.index_to_class.values()))
        y_true = mlb.fit_transform(all_true)
        y_pred = mlb.transform(all_pred)
        hit_rate = sum(
            any(p in true for p in pred)
            for true, pred in zip(all_true, all_pred)
        ) / len(all_true)
        print(f"\n{CYAN}Quick metrics (n={len(all_true)}){RESET}")
        print(f"  Micro F1        : {f1_score(y_true, y_pred, average='micro', zero_division=0):.4f}")
        print(f"  Micro Precision : {precision_score(y_true, y_pred, average='micro', zero_division=0):.4f}")
        print(f"  Micro Recall    : {recall_score(y_true, y_pred, average='micro', zero_division=0):.4f}")
        print(f"  Hit Rate        : {hit_rate:.4f}  (≥1 correct tag per sample)")
    print()


if __name__ == "__main__":
    main()
