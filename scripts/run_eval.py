"""
Evaluate a trained model on the held-out test split.

Usage:
# Local
python scripts/run_eval.py \
--model-dir experiments/run_01 \
--dataset-dir data/processed/tok_scibert_scivocab_uncased \
--batch-size 32 \
--output-dir experiments/run_01


# Kaggle (multi-GPU)
accelerate launch \
--num_processes 2 --num_machines 1 --multi_gpu --mixed_precision fp16 \
scripts/run_eval.py \
--model-dir /kaggle/input/<your-model>/saved_model \
--dataset-dir /kaggle/input/<your-dataset>/tok_scibert-scivocab-uncased \
--batch-size 256 \
--thresholds-file /kaggle/working/thresholds.json \
--output-dir /kaggle/working
"""

import argparse
import json
import logging
import os
import warnings
from functools import partial
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from transformers.utils import logging as hf_logging

from sklearn.metrics import precision_recall_fscore_support

from arxiv_paper_discovery.label_taxonomy import IDX_TO_LABEL
from arxiv_paper_discovery.train import compute_metrics

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)
hf_logging.set_verbosity_error()
logging.getLogger("safetensors").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

GREEN = "\033[92m"
CYAN = "\033[96m"
RESET = "\033[0m"



def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the test split")
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        metavar="DIR",
        help="HF model directory produced by save_pretrained (e.g. saved_models/run_x).",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        metavar="DIR",
        help="Tokenized dataset directory saved via datasets.save_to_disk().",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        metavar="[0-1]",
        help="Sigmoid prediction threshold, used when no thresholds.json is found (default: 0.5).",
    )
    parser.add_argument(
        "--thresholds-file",
        type=Path,
        default=None,
        metavar="FILE",
        help="Per-class thresholds JSON (default: <model-dir>/thresholds.json if it exists).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Where to save test_results.json (default: same as --model-dir).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
        metavar="N",
        help="Evaluation batch size",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir) if args.output_dir else model_dir
    output_path = output_dir / "test_results.json"

    # Resolve per-class thresholds: explicit flag > auto-detect > scalar fallback
    thresholds_file = args.thresholds_file or (model_dir / "thresholds.json")
    per_class_thresholds: np.ndarray | None = None
    if thresholds_file.exists():
        with open(thresholds_file) as f:
            thresholds_dict = json.load(f)
        num_labels = len(IDX_TO_LABEL)
        per_class_thresholds = np.array(
            [thresholds_dict[IDX_TO_LABEL[i]] for i in range(num_labels)], dtype=np.float32
        )

    dataset = load_from_disk(str(args.dataset_dir))
    test_dataset = dataset["test"]

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    eval_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_eval_batch_size=args.batch_size,
        report_to="none",
        disable_tqdm=False,
    )

    if per_class_thresholds is not None:
        def metrics_fn(eval_pred):
            logits, labels = eval_pred
            probs = 1.0 / (1.0 + np.exp(-logits))
            preds = (probs > per_class_thresholds).astype(np.int32)
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, preds, average="macro", zero_division=0
            )
            _, _, f1_weighted, _ = precision_recall_fscore_support(
                labels, preds, average="weighted", zero_division=0
            )
            return {
                "f1": float(f1),
                "f1_weighted": float(f1_weighted),
                "precision": float(precision),
                "recall": float(recall),
            }
    else:
        metrics_fn = partial(compute_metrics, threshold=args.threshold)

    trainer = Trainer(
        model=model,
        args=eval_args,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=metrics_fn,
    )

    print(f"\n{CYAN}{'='*60}{RESET}")
    print(f"{CYAN}Test Set Evaluation{RESET}")
    print(f"  Model      : {model_dir}")
    print(f"  Dataset    : {args.dataset_dir}")
    print(f"  Output     : {output_dir}")
    print(f"  Test count : {len(test_dataset)}")
    print(f"  Batch size : {args.batch_size}")
    if per_class_thresholds is not None:
        print(f"  Thresholds : per-class ({thresholds_file})")
    else:
        print(f"  Threshold  : {args.threshold} (scalar)")
    print(f"{CYAN}{'='*60}{RESET}\n")

    test_results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")

    print(f"{GREEN}Results{RESET}")
    for key, value in test_results.items():
        print(f"  {key:<20}: {value:.4f}")

    with open(output_path, "w") as f:
        json.dump(test_results, f, indent=2)
    print(f"\n{GREEN}Saved -> {output_path}{RESET}\n")


if __name__ == "__main__":
    main()
