"""
Evaluate a Hugging Face saved model on the held-out test split.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from transformers.utils import logging as hf_logging

from arxiv_paper_discovery.train import build_compute_metrics, read_model_threshold

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)
hf_logging.set_verbosity_error()

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
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help="Output JSON file. Defaults to <model-dir>/test_results.json",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        metavar="[0-1]",
        help="Sigmoid prediction threshold. Defaults to model config threshold or 0.5.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
        metavar="N",
        help="Evaluation batch size",
    )
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=1.0,
        metavar="(0-1]",
        help="Fraction of test set to evaluate",
    )
    args = parser.parse_args()

    if not (0.0 < args.sample_ratio <= 1.0):
        raise ValueError("--sample-ratio must be in the range (0, 1].")

    checkpoint_dir = Path(args.model_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {checkpoint_dir}")
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    output_path = args.output or (checkpoint_dir / "test_results.json")

    dataset = load_from_disk(str(dataset_dir))
    test_dataset = dataset["test"]
    if args.sample_ratio < 1.0:
        sample_size = max(1, int(len(test_dataset) * args.sample_ratio))
        test_dataset = test_dataset.select(range(sample_size))

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
    threshold = read_model_threshold(model, args.threshold)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    eval_tmp_dir = Path("outputs") / "_eval_tmp" / checkpoint_dir.name
    eval_args = TrainingArguments(
        output_dir=str(eval_tmp_dir),
        per_device_eval_batch_size=args.batch_size,
        report_to="none",
        disable_tqdm=False,
    )
    trainer = Trainer(
        model=model,
        args=eval_args,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=build_compute_metrics(float(threshold)),
    )

    print(f"\n{CYAN}{'='*60}{RESET}")
    print(f"{CYAN}Test Set Evaluation{RESET}")
    print(f"  Model      : {checkpoint_dir}")
    print(f"  Dataset    : {dataset_dir}")
    print(f"  Test count : {len(test_dataset)}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  Threshold  : {threshold}")
    print(f"{CYAN}{'='*60}{RESET}\n")

    raw_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
    test_results = raw_metrics

    print(f"{GREEN}Results{RESET}")
    for key, value in test_results.items():
        print(f"  {key:<20}: {value:.4f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(test_results, f, indent=2)
    print(f"\n{GREEN}Saved -> {output_path}{RESET}\n")


if __name__ == "__main__":
    main()
