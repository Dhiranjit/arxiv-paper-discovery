"""
Train the SciBERT multi-label classifier on the tokenized ArXiv taxonomy dataset.

This script:
1. Loads training configuration from a YAML file.
2. Resolves dataset and label-map paths from CLI, config, or sensible defaults.
3. Loads tokenized train/validation splits from disk.
4. Builds dataloaders, model, optimizer, scheduler, and loss.
5. Trains with Hugging Face Accelerate (supports mixed precision and accumulation).
6. Tracks validation metrics and saves best/last/step checkpoints (configurable).
7. Supports resuming from a saved checkpoint directory.

Inputs:
- Config file passed via `--config`
- Tokenized dataset directory (CLI/config/default inferred from model name)
- Label index mapping JSON (CLI/config/default)

Output:
- Training artifacts under `saved_models/<run-id>/` (or custom `--save-dir`)

Run:
- `python3 scripts/run_training.py \
    --config <config_path> \
    --run-id <run_id> \
    [--save-dir <save_dir>] \
    [--resume <checkpoint_dir>] \
    [--tokenized-data-dir <dataset_dir>] \
    [--label-map-path <group_to_index.json>] \
    [--no-save]`
    
- `python3 scripts/run_training.py \
    --config configs/scibert_classification.yaml \
    --run-id scibert_baseline`


python3 scripts/run_training.py \
  --config configs/scibert_classification.yaml \
  --run-id scibert_final \
  --sample-ratio 0.01


- `python3 scripts/run_training.py \
    --config configs/scibert_classification.yaml \
    --resume saved_models/<run-id>/last`
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
from transformers.utils import logging
warnings.filterwarnings("ignore", category=FutureWarning)
logging.set_verbosity_error()

import argparse
import json
import math
import re
from pathlib import Path

import torch
import yaml
from accelerate import Accelerator
from datasets import load_from_disk
from transformers import BertModel, BertTokenizer

from arxiv_paper_discovery.config import PROCESSED_DATA_DIR
from arxiv_paper_discovery.data import create_dataloader
from arxiv_paper_discovery.models import SciBERTClassifier
from arxiv_paper_discovery.train import CheckpointConfig, load_checkpoint, train
from arxiv_paper_discovery.utils import build_scheduler, metric_fn, set_seed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_run_id_from_resume_path(resume_path: Path) -> str | None:
    """Infer run id from a resumable checkpoint path: .../<run-id>/(last|step-N)."""
    tag = resume_path.name
    if tag == "last" or tag.startswith("step-"):
        run_id = resume_path.parent.name
        return run_id or None
    return None


def _model_name_to_slug(name: str) -> str:
    """Convert a model id to a filesystem-safe suffix."""
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", name.strip())


def _resolve_tokenized_dataset_path(args: argparse.Namespace, cfg: dict) -> Path:
    """
    Resolve tokenized dataset path in priority order:
    1) CLI `--tokenized-data-dir`
    2) Config `data.tokenized_dataset_dir`
    3) Inferred default: `data/processed/tok_<model-slug>`
    """
    cli_path = getattr(args, "tokenized_data_dir", None)
    if cli_path:
        return Path(cli_path)

    data_cfg = cfg.get("data") or {}
    cfg_path = data_cfg.get("tokenized_dataset_dir")
    if cfg_path:
        return Path(cfg_path)

    model_slug = _model_name_to_slug(cfg["model"]["pretrained_name"])
    return PROCESSED_DATA_DIR / f"tok_{model_slug}"


def _resolve_label_map_path(args: argparse.Namespace, cfg: dict) -> Path:
    """
    Resolve label map path in priority order:
    1) CLI `--label-map-path`
    2) Config `data.group_to_index_path`
    3) Default: `data/processed/group_to_index.json`
    """
    cli_path = getattr(args, "label_map_path", None)
    if cli_path:
        return Path(cli_path)

    data_cfg = cfg.get("data") or {}
    cfg_path = data_cfg.get("group_to_index_path")
    if cfg_path:
        return Path(cfg_path)

    return PROCESSED_DATA_DIR / "group_to_index.json"


# ---------------------------------------------------------------------------
# Core training run
# ---------------------------------------------------------------------------

def execute_training_run(args: argparse.Namespace, cfg: dict) -> dict:
    exp_cfg   = cfg["experiment"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    opt_cfg   = cfg["optimizer"]
    sched_cfg = cfg.get("scheduler") or {}

    accumulation_steps = train_cfg.get("accumulation_steps", 1)
    metric_mode        = train_cfg.get("metric_mode", "max")
    cli_sample_ratio   = getattr(args, "sample_ratio", None)
    sample_ratio       = cli_sample_ratio if cli_sample_ratio is not None else train_cfg.get("sample_ratio")
    threshold          = train_cfg.get("threshold", 0.5)

    ckpt_cfg = CheckpointConfig(
        save_dir          = args.save_dir / args.run_id,
        save_best         = not args.no_save and train_cfg.get("save_best_model", True),
        save_last         = not args.no_save and train_cfg.get("save_last_checkpoint", True),
        every_n_steps     = None if args.no_save else train_cfg.get("save_every_n_steps"),
        keep_n_step_ckpts = train_cfg.get("keep_last_n_step_ckpts", 1),
    )

    set_seed(exp_cfg["seed"])

    # ---- Accelerate ----
    accelerator        = Accelerator(gradient_accumulation_steps=accumulation_steps)
    resolved_precision = accelerator.mixed_precision

    # ---- Data ----
    dataset_path  = _resolve_tokenized_dataset_path(args, cfg)
    label_map_path = _resolve_label_map_path(args, cfg)

    dataset       = load_from_disk(str(dataset_path))
    train_dataset = dataset["train"]
    val_dataset   = dataset["val"]

    if sample_ratio is not None and 0.0 < sample_ratio < 1.0:
        train_limit   = int(len(train_dataset) * sample_ratio)
        val_limit     = int(len(val_dataset)   * sample_ratio)
        train_dataset = train_dataset.shuffle(seed=exp_cfg["seed"]).select(range(train_limit))
        val_dataset   = val_dataset.shuffle(seed=exp_cfg["seed"]).select(range(val_limit))
        accelerator.print(
            f"Sampled {train_limit} train / {val_limit} val examples ({sample_ratio*100:.0f}%)."
        )

    with open(label_map_path) as f:
        class_to_index = json.load(f)

    tokenizer        = BertTokenizer.from_pretrained(model_cfg["pretrained_name"])
    train_dataloader = create_dataloader(
        train_dataset, tokenizer, batch_size=train_cfg["batch_size"], shuffle=True
    )
    val_dataloader   = create_dataloader(
        val_dataset, tokenizer, batch_size=train_cfg["batch_size"], shuffle=False
    )

    # ---- Model ----
    scibert = BertModel.from_pretrained(model_cfg["pretrained_name"])
    model   = SciBERTClassifier(
        llm=scibert, dropout_p=model_cfg["dropout_p"], num_classes=len(class_to_index)
    )

    # ---- Optimizer / scheduler / loss ----
    loss_fn       = torch.nn.BCEWithLogitsLoss()
    optimizer_cls = getattr(torch.optim, opt_cfg["name"])
    optimizer     = optimizer_cls(
        model.parameters(),
        **{k: v for k, v in opt_cfg.items() if k != "name"},
    )
    effective_steps_per_epoch = math.ceil(len(train_dataloader) / accumulation_steps)
    scheduler = build_scheduler(sched_cfg, optimizer, train_cfg["epochs"], effective_steps_per_epoch)

    # ---- Accelerate.prepare ----
    model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, scheduler
    )

    # ---- Resume ----
    resume_state = None
    if args.resume:
        resume_state = load_checkpoint(accelerator, args.resume)

    # ---- Banner ----
    world_size           = accelerator.num_processes
    effective_batch_size = train_cfg["batch_size"] * accumulation_steps * world_size
    scheduler_name       = sched_cfg.get("name", "none")
    resume_mode          = f"resume ({args.resume})" if args.resume else "fresh"
    sample_ratio_text    = f"{sample_ratio:.2f}" if sample_ratio is not None else "none"

    accelerator.print(
        f"\n{'='*72}\n"
        f" Run    : {args.run_id} | Config: {args.config.name}\n"
        f" Mode   : {resume_mode} | Seed: {exp_cfg['seed']} | "
        f"Device: {accelerator.device} | World: {world_size}\n"
        f" Data   : train={len(train_dataset)} | val={len(val_dataset)} | "
        f"sample_ratio={sample_ratio_text}\n"
        f" Paths  : tokenized={dataset_path} | labels={label_map_path}\n"
        f" Batch  : per_device={train_cfg['batch_size']} | accum={accumulation_steps} | "
        f"effective={effective_batch_size}\n"
        f" Steps  : batches/epoch={len(train_dataloader)} | "
        f"opt_steps/epoch={effective_steps_per_epoch}\n"
        f" Optim  : {opt_cfg['name']} | lr={opt_cfg['lr']} | "
        f"wd={opt_cfg.get('weight_decay', 0.0)} | sched={scheduler_name}\n"
        f" Train  : epochs={train_cfg['epochs']} | primary={train_cfg['primary_metric']} | "
        f"mode={metric_mode} | amp={resolved_precision} | threshold={threshold}\n"
        f" Save   : best={ckpt_cfg.save_best} | last={ckpt_cfg.save_last} | "
        f"every_n_steps={ckpt_cfg.every_n_steps} | keep_step_ckpts={ckpt_cfg.keep_n_step_ckpts}\n"
        f"{'='*72}\n"
    )

    results = train(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        accelerator=accelerator,
        epochs=train_cfg["epochs"],
        metric_fn=metric_fn,
        primary_metric=train_cfg["primary_metric"],
        mode=metric_mode,
        scheduler=scheduler,
        config=cfg,
        threshold=threshold,
        ckpt_cfg=ckpt_cfg,
        resume_state=resume_state,
    )

    if accelerator.is_main_process:
        best_val_metrics = results.get("best_metrics", {})
        if best_val_metrics:
            metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in best_val_metrics.items())
            accelerator.print(f"\nBest validation metrics: {metrics_str}\n")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train SciBERT article classifier")
    parser.add_argument("--config",   type=Path, required=True,
                        help="Path to the training YAML config")
    parser.add_argument("--run-id",   type=str,
                        help="Experiment identifier (required for fresh runs; "
                             "inferred from --resume path if omitted)")
    parser.add_argument("--no-save",  action="store_true",
                        help="Disable all model/checkpoint saving")
    parser.add_argument("--resume",   type=Path, default=None,
                        help="Path to a resumable checkpoint dir (last/ or step-N/)")
    parser.add_argument("--save-dir", type=Path, default=Path("saved_models"))
    parser.add_argument("--tokenized-data-dir", type=Path, default=None,
                        help="Path to tokenized dataset directory (overrides config/default)")
    parser.add_argument("--label-map-path", type=Path, default=None,
                        help="Path to group_to_index.json (overrides config/default)")
    parser.add_argument("--sample-ratio", type=float, default=None, metavar="(0-1]",
                        help="Override training.sample_ratio from config (set null in config for full-data default)")

    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.resume and not args.run_id:
        inferred = _infer_run_id_from_resume_path(args.resume)
        if inferred:
            args.run_id = inferred

    if not args.run_id:
        parser.error(
            "--run-id is required for fresh runs. If resuming, pass --resume as "
            ".../<run-id>/last or .../<run-id>/step-N to infer run id automatically."
        )
    if args.sample_ratio is not None and not (0.0 < args.sample_ratio <= 1.0):
        parser.error("--sample-ratio must be in the range (0, 1].")

    execute_training_run(args, cfg)


if __name__ == "__main__":
    main()
