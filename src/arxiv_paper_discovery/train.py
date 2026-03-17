"""
src/arxiv_paper_discovery/train.py

This module provides a streamlined, Accelerate-backed training loop optimized for 
multi-label text classification on single or multi-GPU setups.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from accelerate import Accelerator
from pathlib import Path
from tqdm.auto import tqdm
import mlflow
import yaml
import json
import sys
import optuna

# ANSI COLORS
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"

def train_step(
        model: nn.Module,
        dataloader: DataLoader,
        loss_fn: nn.Module,
        optimizer: Optimizer,
        accelerator: Accelerator,
        epoch_index: int,
        total_epochs: int,
        scheduler: LRScheduler | None = None
):
    model.train()
    train_loss = 0.0

    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch [{epoch_index + 1}/{total_epochs}]",
        disable=not accelerator.is_local_main_process
    )

    for batch_idx, batch in enumerate(progress_bar, 1):
        with accelerator.accumulate(model):
            optimizer.zero_grad(set_to_none=True)
            y = batch["labels"]

            y_pred = model(batch)
            loss = loss_fn(y_pred, y)

            accelerator.backward(loss)

            if accelerator.sync_gradients: 
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            if scheduler and accelerator.sync_gradients:
                scheduler.step()
        
        train_loss += loss.item()

        progress_bar.set_postfix(
            loss=f"{train_loss / batch_idx:.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}"
        )
    
    return train_loss / len(dataloader)


def val_step(
        model: nn.Module, 
        dataloader: DataLoader, 
        loss_fn: nn.Module, 
        accelerator: Accelerator, 
        metric_fn=None
):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for batch in dataloader:
            y = batch["labels"]
            y_pred = model(batch)
            
            preds = (torch.sigmoid(y_pred) > 0.5).int()
            
            preds, y = accelerator.gather_for_metrics((preds, y))

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

            val_loss += loss_fn(y_pred, y).item()

    metrics = metric_fn(all_labels, all_preds) if metric_fn is not None else {}
    return val_loss / len(dataloader), metrics


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: Optimizer,
    loss_fn: nn.Module,
    accelerator: Accelerator,
    epochs: int,
    metric_fn,
    primary_metric: str,
    scheduler=None,
    config: dict | None = None,
    mode: str = "max",
    save_best_model: bool = True,
    save_dir: Path | None = None,
    trial: optuna.Trial | None = None
):
    best_val_metric = float("-inf") if mode == "max" else float("inf")
    best_val_metrics_dict = {}
    results = {"train_loss": [], "val_loss": []}

    runtime_params = {
        "epochs": epochs,
        "optimizer": type(optimizer).__name__,
        "scheduler": type(scheduler).__name__ if scheduler else "None",
        "loss_fn": type(loss_fn).__name__,
        "primary_metric": primary_metric,
        "metric_mode": mode,
        "initial_lr": optimizer.param_groups[0]["lr"],
    }
    mlflow.log_params(runtime_params)

    try:
        for epoch in range(epochs):
            train_loss = train_step(
                model, train_dataloader, loss_fn, optimizer, 
                accelerator, epoch, epochs, scheduler=scheduler
            )
            
            val_loss, metrics = val_step(
                model, val_dataloader, loss_fn, accelerator, metric_fn=metric_fn
            )

            # --- Update Metrics ---
            results["train_loss"].append(train_loss)
            results["val_loss"].append(val_loss)
            for k, v in metrics.items():
                results.setdefault(f"val_{k}", []).append(v)

            current_metrics = {"loss": val_loss, **metrics}

            # --- Printing ---
            padding = " " * (len(f"Epoch [{epoch + 1}/{epochs}]") + 2)
            metrics_str = " | ".join(
                f"{CYAN}val_{k}:{RESET} {GREEN}{v:.4f}{RESET}" for k, v in metrics.items()
            )
            accelerator.print(
                f"{padding}{CYAN}train_loss:{RESET} {YELLOW}{train_loss:.4f}{RESET} | "
                f"{CYAN}val_loss:{RESET} {YELLOW}{val_loss:.4f}{RESET}"
                + (f" | {metrics_str}" if metrics_str else "")
            )

            # --- Log Metrics to MLflow ---
            mlflow.log_metrics(
                {"train_loss": train_loss, **{f"val_{k}": v for k, v in current_metrics.items()}},
                step=epoch,
            )
            
            fallback_val = float("inf") if mode == "min" else float("-inf")
            current_metric_val = current_metrics.get(primary_metric, fallback_val)
            
            # --- Optuna Pruning Integration ---
            should_prune = False
            if trial is not None:
                if accelerator.is_main_process:
                    trial.report(current_metric_val, epoch)
                    if trial.should_prune():
                        should_prune = True
                
                # Broadcast prune decision across processes to prevent distributed deadlocks
                prune_tensor = torch.tensor(1 if should_prune else 0, device=accelerator.device)
                prune_tensor = accelerator.reduce(prune_tensor, reduction="max")
                
                if prune_tensor.item() > 0:
                    accelerator.print(f"\n{YELLOW}Trial pruned by Optuna at epoch {epoch}.{RESET}")
                    raise optuna.exceptions.TrialPruned()

            # --- Check if Best ---
            is_best = (current_metric_val < best_val_metric) if mode == "min" else (current_metric_val > best_val_metric)
            
            if is_best:
                best_val_metric = current_metric_val
                best_val_metrics_dict = {"val_loss": val_loss, **metrics}
                
                best_metrics_to_log = {f"best_val_{k}": v for k, v in current_metrics.items()}
                mlflow.log_metrics(best_metrics_to_log, step=epoch)

                # --- Save Best Model Locally ---
                if save_best_model and save_dir:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        save_dir.mkdir(parents=True, exist_ok=True)
                        
                        unwrapped_model = accelerator.unwrap_model(model)
                        torch.save(unwrapped_model.state_dict(), save_dir / "best_model.pth")
                        
                        with open(save_dir / "best_metrics.json", "w") as f:
                            json.dump(best_val_metrics_dict, f, indent=2)
                            
                        if config:
                            with open(save_dir / "config.yaml", "w") as f:
                                yaml.safe_dump(config, f)
                                
                        accelerator.print(f"{GREEN}>>> Best model updated in {save_dir}{RESET}")

    except KeyboardInterrupt:
        accelerator.print(f"\n{YELLOW}Training interrupted by user.{RESET}")
        sys.exit(0)

    accelerator.print(f"{GREEN}Training complete!{RESET}")
    return {"history": results, "best_metrics": best_val_metrics_dict}


def eval_model(model, dataloader, loss_fn, accelerator, metric_fn):
    accelerator.print(f"{CYAN}Evaluating model on test set...{RESET}")
    
    test_loss, metrics = val_step(model, dataloader, loss_fn, accelerator, metric_fn=metric_fn)

    metrics_str = " | ".join(
        f"{CYAN}Test {k}:{RESET} {GREEN}{v:.4f}{RESET}" for k, v in metrics.items()
    )
    
    accelerator.print(
        f"\n{CYAN}Test Loss:{RESET} {YELLOW}{test_loss:.4f}{RESET} | "
        f"{metrics_str}\n"
    )

    return {"loss": test_loss, **metrics}