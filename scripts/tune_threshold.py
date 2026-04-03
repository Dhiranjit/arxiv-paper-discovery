"""
Tune per-class classification thresholds on the validation split.

Runs inference once to get logits, then sweeps thresholds per class to
maximise binary F1. Saves a JSON file with one threshold per label.

Usage:
# Local
python scripts/tune_threshold.py \
--model-dir experiments/run_01 \
--dataset-dir data/processed/tok_scibert_scivocab_uncased \
--batch-size 64

# Kaggle (multi-GPU)
accelerate launch \
--num_processes 2 --num_machines 1 --multi_gpu --mixed_precision fp16 \
scripts/tune_threshold.py \
--model-dir /kaggle/input/<your-model>/saved_model \
--dataset-dir /kaggle/input/<your-dataset>/tok_scibert-scivocab-uncased \
--batch-size 256 \
--output-file /kaggle/working/thresholds.json
"""

import argparse
import json
import logging
import os
import warnings
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from sklearn.metrics import f1_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from transformers.utils import logging as hf_logging

from arxiv_paper_discovery.label_taxonomy import IDX_TO_LABEL, LABELS

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)
hf_logging.set_verbosity_error()
logging.getLogger("safetensors").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RESET = "\033[0m"

SWEEP = np.arange(0.05, 0.955, 0.01)


def tune(probs: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (best_thresholds, best_f1s) arrays of shape [num_labels]."""
    num_labels = probs.shape[1]
    best_thresholds = np.zeros(num_labels)
    best_f1s = np.zeros(num_labels)

    for c in range(num_labels):
        best_t, best_f1 = 0.5, 0.0
        for t in SWEEP:
            preds_c = (probs[:, c] > t).astype(np.int32)
            f1 = f1_score(labels[:, c], preds_c, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        best_thresholds[c] = round(float(best_t), 4)
        best_f1s[c] = round(float(best_f1), 4)

    return best_thresholds, best_f1s


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune per-class thresholds on the validation split")
    parser.add_argument("--model-dir", type=Path, required=True, metavar="DIR")
    parser.add_argument("--dataset-dir", type=Path, required=True, metavar="DIR")
    parser.add_argument("--batch-size", type=int, required=True, metavar="N")
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        metavar="FILE",
        help="Where to save thresholds JSON (default: <model-dir>/thresholds.json)",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_file = args.output_file or model_dir / "thresholds.json"

    dataset = load_from_disk(str(args.dataset_dir))
    val_dataset = dataset["val"]


    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    eval_args = TrainingArguments(
        output_dir=str(output_file.parent),
        per_device_eval_batch_size=args.batch_size,
        report_to="none",
        disable_tqdm=False,
    )
    trainer = Trainer(
        model=model,
        args=eval_args,
        processing_class=tokenizer,
        data_collator=collator,
    )

    print(f"\n{CYAN}{'='*60}{RESET}")
    print(f"{CYAN}Threshold Tuning (val split){RESET}")
    print(f"  Model      : {model_dir}")
    print(f"  Dataset    : {args.dataset_dir}")
    print(f"  Val count  : {len(val_dataset)}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  Sweep      : {SWEEP[0]:.2f} → {SWEEP[-1]:.2f} ({len(SWEEP)} steps)")
    print(f"{CYAN}{'='*60}{RESET}\n")

    print("Running inference on val set...")
    predictions = trainer.predict(val_dataset)
    logits = predictions.predictions        # [N, 25]
    labels = predictions.label_ids          # [N, 25]

    probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid

    print("Sweeping thresholds...\n")
    best_thresholds, best_f1s = tune(probs, labels)

    # Baseline: scalar 0.35
    baseline_preds = (probs > 0.35).astype(np.int32)
    _, _, baseline_f1, _ = precision_recall_fscore_support(
        labels, baseline_preds, average="macro", zero_division=0
    )
    tuned_preds = (probs > best_thresholds).astype(np.int32)
    _, _, tuned_f1, _ = precision_recall_fscore_support(
        labels, tuned_preds, average="macro", zero_division=0
    )

    # Print per-class table
    col_w = max(len(l) for l in LABELS) + 2
    print(f"{'Label':<{col_w}}  {'Threshold':>9}  {'F1':>6}")
    print("-" * (col_w + 20))
    for i in range(len(LABELS)):
        label = IDX_TO_LABEL[i]
        print(f"{label:<{col_w}}  {best_thresholds[i]:>9.4f}  {best_f1s[i]:>6.4f}")

    print(f"\n{YELLOW}Macro F1 — scalar 0.35 : {baseline_f1:.4f}{RESET}")
    print(f"{GREEN}Macro F1 — per-class   : {tuned_f1:.4f}{RESET}")

    # Save
    thresholds_dict = {IDX_TO_LABEL[i]: float(best_thresholds[i]) for i in range(len(LABELS))}
    with open(output_file, "w") as f:
        json.dump(thresholds_dict, f, indent=2)
    print(f"\n{GREEN}Saved -> {output_file}{RESET}\n")


if __name__ == "__main__":
    main()
