"""
Train the multi-label paper classifier with Hugging Face Trainer.

Config is the source of truth for model and training hyperparameters.
Dataset path and output directory are passed via CLI so the same config
works unchanged across local and Kaggle/Colab environments.

Usage:
accelerate launch scripts/run_training.py \
--config configs/scibert.yaml \
--dataset-path data/processed/tok_scibert_scivocab_uncased \
--output-dir experiments/run_01 \
--sample-ratio 0.01


# Resume from a checkpoint:
accelerate launch scripts/run_training.py \
--config configs/scibert.yaml \
--dataset-path data/processed/tok_scibert_scivocab_uncased \
--output-dir experiments/run_01 \
--resume-dir experiments/run_01/checkpoint-6000


# Running in Kaggle (multi-GPU):
accelerate launch \
--num_processes 2 --num_machines 1 --multi_gpu --mixed_precision fp16 \
scripts/run_training.py \
--config configs/scibert_kaggle.yaml \
--dataset-path /kaggle/input/<your-dataset>/tok_scibert-scivocab-uncased \
--output-dir /kaggle/working/saved_models
"""


import argparse
import logging
import os
import warnings
from pathlib import Path

import yaml
from transformers.utils import logging as hf_logging

from arxiv_paper_discovery.train import train

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)
hf_logging.set_verbosity_error()
logging.getLogger("safetensors").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the paper classifier with HF Trainer")
    parser.add_argument("--config", type=Path, required=True, help="Path to training YAML config")
    parser.add_argument("--dataset-path", type=Path, required=True, help="Path to tokenized HF dataset")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save model and artifacts")
    parser.add_argument("--resume-dir", type=Path, default=None, help="Checkpoint directory to resume from (optional)")
    parser.add_argument("--sample-ratio", type=float, default=None, help="Override config sample_ratio (e.g. 0.01 for quick test)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}

    if args.sample_ratio is not None:
        cfg.setdefault("training", {})["sample_ratio"] = args.sample_ratio

    print(f"Running training with config: {args.config}")
    results = train(cfg, dataset_path=args.dataset_path, output_dir=args.output_dir,
                    resume_from_checkpoint=args.resume_dir)

    eval_metrics = results.get("eval_metrics", {})
    if eval_metrics:
        metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in eval_metrics.items())
        print(f"\nValidation metrics: {metrics_str}\n")


if __name__ == "__main__":
    main()
