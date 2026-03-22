"""
Computes the official test-set metrics using eval_model() from train.py.

Input  : tokenized test split (tok_scibert_scivocab_uncased/test)
Output : test_results.json containing metrics like F1, precision, recall, etc.

Usage:
python scripts/eval_test_set.py \
    --model-dir saved_models/my_run/best \
    --batch-size 64 \
    --threshold 0.35 \
    --sample-ratio 0.1
    

"""

import os
import argparse
import json
import warnings
from pathlib import Path

import torch
import yaml
from accelerate import Accelerator
from datasets import load_from_disk
from transformers import BertModel, BertTokenizer

# Suppress warnings from tokenizers and transformers (e.g. about parallelism, future deprecations, etc.)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers.utils import logging
warnings.filterwarnings("ignore", category=FutureWarning)
logging.set_verbosity_error()

from arxiv_paper_discovery.config import PROCESSED_DATA_DIR
from arxiv_paper_discovery.data import create_dataloader
from arxiv_paper_discovery.models import SciBERTClassifier
from arxiv_paper_discovery.train import eval_model
from arxiv_paper_discovery.utils import metric_fn

GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on the held-out test set."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        metavar="DIR",
        help="Path to the checkpoint directory 'saved_model/run-id'",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help="Output JSON file. Defaults to 'saved_model/run-id/best/test_results.json'",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        metavar="[0-1]",
        help="Sigmoid prediction threshold. Overides config value",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
        metavar="N",
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=1.0,
        metavar="(0-1]",
        help="Fraction of the test set to evaluate.",
    )

    args = parser.parse_args()

    checkpoint_dir = Path(args.model_dir)
    model_path     = checkpoint_dir / "model.pth"
    config_path    = checkpoint_dir / "config.yaml"
    output_path    = args.output or (checkpoint_dir / "test_results.json")


    for p, label in [(model_path, "model.pth"), (config_path, "config.yaml")]:
        if not p.exists():
            raise FileNotFoundError(
                f"{label} not found in '{checkpoint_dir}'. "
                "Point --checkpoint at a best/ directory from training."
            )

    # ---- Load config ----
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_cfg  = cfg["model"]
    train_cfg  = cfg.get("training", {})
    batch_size = args.batch_size
    sample_ratio = args.sample_ratio
    threshold  = args.threshold if args.threshold is not None else train_cfg.get("threshold", 0.5)

    if not (0.0 < sample_ratio <= 1.0):
        raise ValueError("--sample-ratio must be in the range (0, 1].")


    accelerator = Accelerator()
    accelerator.print(f"\n{CYAN}{'='*60}{RESET}")
    accelerator.print(f"{CYAN}Test Set Evaluation{RESET}")
    accelerator.print(f"  Checkpoint : {checkpoint_dir}")
    accelerator.print(f"  Device     : {accelerator.device}")
    accelerator.print(f"  Batch size : {batch_size}")
    accelerator.print(f"  SampleRatio: {sample_ratio}")
    accelerator.print(f"  Threshold  : {threshold}")
    accelerator.print(f"{CYAN}{'='*60}{RESET}\n")

    # ---- Tokenized test split ----
    dataset_path = PROCESSED_DATA_DIR / "tok_scibert_scivocab_uncased"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Tokenized dataset not found at {dataset_path}. "
            "Run the preprocessing pipeline first."
        )

    dataset      = load_from_disk(str(dataset_path))
    test_dataset = dataset["test"]

    if sample_ratio < 1.0:
        sample_size = max(1, int(len(test_dataset) * sample_ratio))
        test_dataset = test_dataset.select(range(sample_size))

    accelerator.print(f"Test examples : {len(test_dataset)}")

    with open(PROCESSED_DATA_DIR / "group_to_index.json") as f:
        class_to_index = json.load(f)

    tokenizer       = BertTokenizer.from_pretrained(model_cfg["pretrained_name"])
    test_dataloader = create_dataloader(
        test_dataset, tokenizer, batch_size=batch_size, shuffle=False
    )

    # ---- Model ----
    scibert = BertModel.from_pretrained(model_cfg["pretrained_name"])
    model   = SciBERTClassifier(
        llm=scibert,
        dropout_p=model_cfg["dropout_p"],
        num_classes=len(class_to_index),
    )

    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)

    # ---- Accelerate.prepare (moves model + dataloader to the right device) ----
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model, test_dataloader = accelerator.prepare(model, test_dataloader)

    # ---- Evaluate ----
    accelerator.print(f"{CYAN}Running eval...{RESET}")
    test_results = eval_model(
        model=model,
        dataloader=test_dataloader,
        loss_fn=loss_fn,
        accelerator=accelerator,
        metric_fn=metric_fn,
        threshold=threshold,
    )

    # ---- Print & save ----
    if accelerator.is_main_process:
        accelerator.print(f"\n{GREEN}Results{RESET}")
        for k, v in test_results.items():
            accelerator.print(f"  {k:<20}: {v:.4f}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(test_results, f, indent=2)

        accelerator.print(f"\n{GREEN}Saved → {output_path}{RESET}\n")


if __name__ == "__main__":
    main()