import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
from transformers.utils import logging
warnings.filterwarnings("ignore", category=FutureWarning)
logging.set_verbosity_error()

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

import mlflow
import torch
import yaml
import optuna
from datasets import load_from_disk
from accelerate import Accelerator
from transformers import BertModel, BertTokenizer

from arxiv_paper_discovery.config import PROCESSED_DATA_DIR
from arxiv_paper_discovery.data import create_dataloader
from arxiv_paper_discovery.models import SciBERTClassifier
from arxiv_paper_discovery.train import train
from arxiv_paper_discovery.utils import metric_fn, set_seed, build_scheduler, flatten_dict


def _set_nested(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    cursor = cfg
    for key in keys[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[keys[-1]] = value


def _apply_sweep_space(trial: optuna.trial.Trial, cfg: dict[str, Any], sweep_cfg: dict[str, Any] | None) -> dict[str, Any]:
    sampled_params: dict[str, Any] = {}

    sweep_params = (sweep_cfg or {}).get("sweep", {})
    if not sweep_params:
        # Backward-compatible fallback when no external sweep config is provided.
        lr = trial.suggest_float("optimizer.lr", 1e-5, 1e-4, log=True)
        dropout_p = trial.suggest_categorical("model.dropout_p", [0.1, 0.2, 0.3, 0.4, 0.5])
        _set_nested(cfg, "optimizer.lr", lr)
        _set_nested(cfg, "model.dropout_p", dropout_p)
        sampled_params["optimizer.lr"] = lr
        sampled_params["model.dropout_p"] = dropout_p
        return sampled_params

    for name, spec in sweep_params.items():
        param_type = spec.get("type", "float")
        if param_type == "float":
            sampled_value = trial.suggest_float(
                name,
                spec["low"],
                spec["high"],
                log=spec.get("log", False),
                step=spec.get("step")
            )
        elif param_type == "int":
            sampled_value = trial.suggest_int(
                name,
                spec["low"],
                spec["high"],
                log=spec.get("log", False),
                step=spec.get("step", 1)
            )
        elif param_type == "categorical":
            sampled_value = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"Unsupported sweep param type '{param_type}' for '{name}'.")

        _set_nested(cfg, name, sampled_value)
        sampled_params[name] = sampled_value

    return sampled_params


def execute_training_run(args, cfg, trial=None):
    exp_cfg = cfg["experiment"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    opt_cfg = cfg["optimizer"]
    sched_cfg = cfg.get("scheduler", {}) or {}

    accumulation_steps = train_cfg.get("accumulation_steps", 1)
    metric_mode = train_cfg.get("metric_mode", "max")
    
    max_train_samples = train_cfg.get("max_train_samples")
    max_val_samples = train_cfg.get("max_val_samples")

    set_seed(exp_cfg["seed"])

    # --- Initialize Accelerate ---
    accelerator = Accelerator(
        gradient_accumulation_steps=accumulation_steps
    )
    resolved_precision = accelerator.mixed_precision

    dataset = load_from_disk(str(PROCESSED_DATA_DIR / "arxiv_classification_dataset"))
    train_dataset = dataset["train"]
    val_dataset = dataset["val"]

    if max_train_samples is not None:
        limit = min(max_train_samples, len(train_dataset))
        train_dataset = train_dataset.select(range(limit))
        accelerator.print(f"Limiting training dataset to {limit} samples.")

    if max_val_samples is not None:
        limit = min(max_val_samples, len(val_dataset))
        val_dataset = val_dataset.select(range(limit))
        accelerator.print(f"Limiting validation dataset to {limit} samples.")

    with open(PROCESSED_DATA_DIR / "category_to_index.json") as f:
        class_to_index = json.load(f)

    tokenizer = BertTokenizer.from_pretrained(model_cfg["pretrained_name"])

    train_dataloader = create_dataloader(
        train_dataset, tokenizer, batch_size=train_cfg["batch_size"], shuffle=True
    )
    val_dataloader = create_dataloader(
        val_dataset, tokenizer, batch_size=train_cfg["batch_size"], shuffle=False
    )

    scibert = BertModel.from_pretrained(model_cfg["pretrained_name"])
    model = SciBERTClassifier(
        llm=scibert, dropout_p=model_cfg["dropout_p"], num_classes=len(class_to_index)
    )

    loss_fn = torch.nn.BCEWithLogitsLoss()

    optimizer_cls = getattr(torch.optim, opt_cfg["name"])
    opt_kwargs = {k: v for k, v in opt_cfg.items() if k != "name"}
    optimizer = optimizer_cls(model.parameters(), **opt_kwargs)

    effective_steps_per_epoch = math.ceil(len(train_dataloader) / accumulation_steps)
    scheduler = build_scheduler(
        sched_cfg, optimizer, train_cfg["epochs"], effective_steps_per_epoch
    )

    model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, scheduler
    )

    run_save_dir = args.save_dir / args.run_id

    if accelerator.is_main_process:
        args.mlflow_dir.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(f"file://{args.mlflow_dir.resolve()}")
        mlflow.set_experiment(exp_cfg["name"])
        
        mlflow.start_run(run_name=args.run_id)
        
        mlflow.log_artifact(str(args.config), artifact_path="config")
        flat_cfg = flatten_dict(cfg)
        flat_cfg["hardware/mixed_precision"] = resolved_precision
        mlflow.log_params(flat_cfg)

    accelerator.print(f"\n{'='*60}")
    accelerator.print(f" Run: {args.run_id} | Config: {args.config.name}")
    accelerator.print(f" bs: {train_cfg['batch_size']} | accum: {accumulation_steps} | "
          f"lr: {opt_cfg['lr']} | epochs: {train_cfg['epochs']} | amp: {resolved_precision}")
    accelerator.print(f"{'='*60}\n")

    try:
        results = train(
            model=model, train_dataloader=train_dataloader, val_dataloader=val_dataloader,
            optimizer=optimizer, loss_fn=loss_fn, accelerator=accelerator,
            epochs=train_cfg["epochs"], metric_fn=metric_fn,             
            primary_metric=train_cfg["primary_metric"], mode=metric_mode,
            scheduler=scheduler, config=cfg, save_best_model=not args.no_save,
            save_dir=run_save_dir, trial=trial
        )
    except optuna.exceptions.TrialPruned:
        if accelerator.is_main_process:
            mlflow.end_run(status="KILLED")
        raise

    if accelerator.is_main_process:
        best_val_metrics = results.get("best_metrics", {})
        if best_val_metrics:
            metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in best_val_metrics.items())
            accelerator.print(f"\nBest validation metrics: {metrics_str}\n")
            
            run_save_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = run_save_dir / "metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(best_val_metrics, f)

        mlflow.end_run()

    return results


def sweep_objective(trial, args, base_cfg, sweep_cfg=None):
    """Callback for running an isolated trial locally inside the subprocess worker."""
    cfg = copy.deepcopy(base_cfg)
    sampled_params = _apply_sweep_space(trial, cfg, sweep_cfg)
    sampled_tokens = [f"{k.split('.')[-1]}-{v}" for k, v in sampled_params.items()]
    args.run_id = f"sweep-trial-{trial.number}_" + "_".join(sampled_tokens)
    
    results = execute_training_run(args, cfg, trial=trial)
    
    primary_metric = cfg.get("training", {}).get("primary_metric", "f1")
    best_score = results.get("best_metrics", {}).get(primary_metric)
    
    if best_score is None:
        raise ValueError(f"Could not find '{primary_metric}' in results for Trial {trial.number}.")
        
    return best_score


def main():
    parser = argparse.ArgumentParser(description="Train SciBERT article classifier")
    parser.add_argument("--config", type=Path, required=True, help="Path to the YAML config")
    parser.add_argument("--sweep-config", type=Path, default=Path("configs/scibert_sweep.yaml"),
                        help="Path to YAML sweep definition (used with --sweep)")
    parser.add_argument("--run-id", type=str, help="Experiment identifier (required if not sweeping)")
    parser.add_argument("--no-save", action="store_true", help="Disable saving model weights")
    parser.add_argument("--save-dir", type=Path, default=Path("saved_models"))
    parser.add_argument("--mlflow-dir", type=Path, default=Path("mlruns"))
    
    # Optuna integration arguments
    parser.add_argument("--sweep", action="store_true", help="Run as part of an Optuna sweep via SQLite.")
    parser.add_argument("--study-name", type=str, default="scibert-sweep")
    parser.add_argument("--storage", type=str, default="sqlite:///scibert_sweep.db")
    
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    sweep_cfg = None
    if args.sweep and args.sweep_config.exists():
        with open(args.sweep_config) as f:
            sweep_cfg = yaml.safe_load(f) or {}

    if args.sweep:
        study = optuna.load_study(study_name=args.study_name, storage=args.storage)
        # Execute exactly one trial per subprocess
        study.optimize(lambda t: sweep_objective(t, args, cfg, sweep_cfg=sweep_cfg), n_trials=1)
    else:
        if not args.run_id:
            parser.error("--run-id is required when not running a sweep.")
        execute_training_run(args, cfg)


if __name__ == "__main__":
    main()