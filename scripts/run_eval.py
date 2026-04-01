"""
Evaluate a trained model on the held-out test split.

Usage:
python scripts/run_eval.py \
--model-dir experiments/run_01 \
--dataset-dir data/processed/tok_scibert_scivocab_uncased \
--batch-size 32
"""

import argparse
import json
import logging
import os
import warnings
from functools import partial
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
        help="Sigmoid prediction threshold (default: 0.5).",
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
    output_path = model_dir / "test_results.json"

    dataset = load_from_disk(str(args.dataset_dir))
    test_dataset = dataset["test"]

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    eval_args = TrainingArguments(
        output_dir=str(model_dir),
        per_device_eval_batch_size=args.batch_size,
        report_to="none",
        disable_tqdm=False,
    )
    trainer = Trainer(
        model=model,
        args=eval_args,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=partial(compute_metrics, threshold=args.threshold),
    )

    print(f"\n{CYAN}{'='*60}{RESET}")
    print(f"{CYAN}Test Set Evaluation{RESET}")
    print(f"  Model      : {model_dir}")
    print(f"  Dataset    : {args.dataset_dir}")
    print(f"  Test count : {len(test_dataset)}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  Threshold  : {args.threshold}")
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
