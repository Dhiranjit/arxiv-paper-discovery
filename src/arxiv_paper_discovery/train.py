"""
src/arxiv_paper_discovery/train.py

Accelerate-backed training loop for multi-label text classification.
Supports epoch- and step-based checkpointing, best-model saving, and resume.

Checkpoint layout
-----------------
  saved_models/{run-id}/
    best/               ← inference checkpoint: model.pth + metrics.json + config.yaml
    last/               ← resumable: accelerator state + resume.json + config.yaml
    step-{N}/           ← resumable: same layout as last/, written every N opt steps
    train_history.json  ← per-epoch train/val loss + metrics, appended each epoch
    results.json        ← best metrics written once training completes

Resumable checkpoints (last/ and step-N/) use accelerator.save_state() which bundles
the model, optimizer, scheduler, RNG state, and AMP GradScaler in one shot.
resume.json stores the loop counters needed to continue training:
  { "epoch": int, "global_step": int, "best_val_metric": float }

best/ uses torch.save(state_dict) only — it is an inference artefact, not resumable.
"""

import json
import math
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml
from accelerate import Accelerator
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# ANSI COLORS
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"


# ---------------------------------------------------------------------------
# CheckpointConfig
# ---------------------------------------------------------------------------

@dataclass
class CheckpointConfig:
    """Bundles all checkpointing knobs so train() doesn't need 6 extra params."""
    save_dir:          Path | None = None
    save_best:         bool        = True
    save_last:         bool        = True
    every_n_steps:     int | None  = None
    keep_n_step_ckpts: int         = 1


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_better(current: float, best: float, mode: str) -> bool:
    return current > best if mode == "max" else current < best


def _save_best(
    accelerator: Accelerator,
    save_dir: Path,
    model: nn.Module,
    metrics: dict,
    config: dict | None,
) -> None:
    """
    Inference-only checkpoint. Saves model.pth + metrics.json + config.yaml.
    Not resumable — does not include optimizer/scheduler state.
    """
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return

    ckpt_dir = save_dir / "best"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.save(accelerator.unwrap_model(model).state_dict(), ckpt_dir / "model.pth")

    with open(ckpt_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    if config:
        with open(ckpt_dir / "config.yaml", "w") as f:
            yaml.safe_dump(config, f)

    accelerator.print(f"{GREEN}>>> Best checkpoint saved → {ckpt_dir}{RESET}")


def _save_resumable(
    accelerator: Accelerator,
    save_dir: Path,
    tag: str,
    resume_state: dict,
    config: dict | None,
    *,
    keep_last_n: int = 1,
) -> None:
    """
    Resumable checkpoint using accelerator.save_state().
    Writes to save_dir/tag/ and prunes older step checkpoints if keep_last_n is set.

    resume_state: { "epoch": int, "global_step": int, "best_val_metric": float }
    keep_last_n:  how many step-N/ directories to keep (only applied to step-* tags).
                  last/ is always a single overwrite so pruning is not needed there.
    """
    accelerator.wait_for_everyone()

    ckpt_dir = save_dir / tag
    accelerator.save_state(output_dir=str(ckpt_dir))

    if accelerator.is_main_process:
        with open(ckpt_dir / "resume.json", "w") as f:
            json.dump(resume_state, f, indent=2)

        if config:
            with open(ckpt_dir / "config.yaml", "w") as f:
                yaml.safe_dump(config, f)

        accelerator.print(f"{GREEN}>>> Resumable checkpoint '{tag}' saved → {ckpt_dir}{RESET}")

        # Prune old step checkpoints
        if tag.startswith("step-") and keep_last_n > 0:
            step_dirs = sorted(
                save_dir.glob("step-*"),
                key=lambda p: int(p.name.split("-")[1]),
            )
            for old_dir in step_dirs[:-keep_last_n]:
                shutil.rmtree(old_dir, ignore_errors=True)
                accelerator.print(f"{YELLOW}>>> Pruned old checkpoint {old_dir.name}{RESET}")


def _append_epoch_history(save_dir: Path, record: dict) -> None:
    """Append one epoch's metrics to train_history.json (list of dicts, one per epoch)."""
    history_path = save_dir / "train_history.json"
    history: list[dict] = []
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
    history.append(record)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)


def _write_results(save_dir: Path, results: dict) -> None:
    """Write final best metrics to results.json."""
    with open(save_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)


def load_checkpoint(accelerator: Accelerator, resume_dir: Path) -> dict:
    """
    Restores full training state from a resumable checkpoint directory.
    Must be called after accelerator.prepare() so all objects are wrapped.

    Returns the resume_state dict:
        { "epoch": int, "global_step": int, "best_val_metric": float }
    """
    resume_json = resume_dir / "resume.json"
    if not resume_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {resume_dir}")
    if not resume_json.exists():
        raise FileNotFoundError(
            f"{resume_dir} does not contain resume.json — "
            "this looks like a best/ checkpoint which is not resumable. "
            "Point --resume at a last/ or step-N/ directory instead."
        )

    accelerator.load_state(str(resume_dir))

    with open(resume_json) as f:
        resume_state = json.load(f)

    accelerator.print(
        f"{CYAN}Resumed from '{resume_dir.name}' "
        f"(epoch {resume_state['epoch']}, step {resume_state['global_step']}){RESET}"
    )
    return resume_state


# ---------------------------------------------------------------------------
# train_step
# ---------------------------------------------------------------------------

def train_step(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    accelerator: Accelerator,
    epoch_index: int,
    total_epochs: int,
    scheduler: LRScheduler | None = None,
    ckpt_cfg: CheckpointConfig | None = None,
    global_step: int = 0,
    resume_state: dict | None = None,
    config: dict | None = None,
    best_val_metric: float = float("-inf"),
) -> tuple[float, int]:
    """
    Runs one epoch of training.

    When resuming mid-epoch (resume_state contains a global_step that falls
    inside this epoch), batches up to the already-completed step are skipped
    via accelerator.skip_first_batches().

    Returns:
        (avg_train_loss, updated_global_step)
    """
    model.train()
    train_loss = 0.0

    ckpt_cfg = ckpt_cfg or CheckpointConfig()

    # --- Mid-epoch resume: skip batches already seen before the crash ---
    steps_per_epoch = len(dataloader)
    accum_steps     = accelerator.gradient_accumulation_steps

    if resume_state is not None:
        # Must mirror optimizer-step accounting used during training:
        # one optimizer step is taken at each gradient-sync boundary, i.e. ceil(batches/accum).
        opt_steps_per_epoch         = math.ceil(steps_per_epoch / accum_steps)
        steps_before_epoch          = resume_state["epoch"] * opt_steps_per_epoch
        completed_steps_this_epoch  = resume_state["global_step"] - steps_before_epoch
        if completed_steps_this_epoch > 0:
            batches_to_skip = completed_steps_this_epoch * accum_steps
            dataloader = accelerator.skip_first_batches(dataloader, batches_to_skip)
            accelerator.print(
                f"{YELLOW}Skipping {batches_to_skip} batches already trained "
                f"({completed_steps_this_epoch} opt steps){RESET}"
            )

    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch [{epoch_index + 1}/{total_epochs}]",
        disable=not accelerator.is_local_main_process,
        dynamic_ncols=True,
        leave=False,
    )

    for batch_idx, batch in enumerate(progress_bar, 1):
        with accelerator.accumulate(model):
            optimizer.zero_grad(set_to_none=True)

            y      = batch["labels"].float()
            y_pred = model(batch)
            loss   = loss_fn(y_pred, y)

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            if accelerator.sync_gradients:
                if scheduler:
                    scheduler.step()
                global_step += 1

                # --- Step-level resumable checkpoint ---
                if (
                    ckpt_cfg.every_n_steps
                    and ckpt_cfg.save_dir
                    and global_step % ckpt_cfg.every_n_steps == 0
                ):
                    _save_resumable(
                        accelerator, ckpt_cfg.save_dir,
                        tag=f"step-{global_step}",
                        resume_state={
                            "epoch":           epoch_index,
                            "global_step":     global_step,
                            "best_val_metric": best_val_metric,
                        },
                        config=config,
                        keep_last_n=ckpt_cfg.keep_n_step_ckpts,
                    )

        train_loss += loss.item()
        progress_bar.set_postfix(
            loss=f"{train_loss / batch_idx:.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )

    return train_loss / len(dataloader), global_step


# ---------------------------------------------------------------------------
# val_step
# ---------------------------------------------------------------------------

def val_step(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    accelerator: Accelerator,
    metric_fn=None,
    threshold: float = 0.5,
) -> tuple[float, dict]:
    model.eval()
    val_loss   = 0.0
    all_preds  = []
    all_labels = []

    progress_bar = tqdm(
        dataloader,
        desc="Validation",
        disable=not accelerator.is_local_main_process,
        dynamic_ncols=True,
        leave=False,
    )

    with torch.inference_mode():
        for batch in progress_bar:
            y      = batch["labels"].float()
            y_pred = model(batch)

            preds    = (torch.sigmoid(y_pred) > threshold).int()
            preds, y = accelerator.gather_for_metrics((preds, y))

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

            val_loss += loss_fn(y_pred, y).item()

    metrics = metric_fn(all_labels, all_preds) if metric_fn is not None else {}
    return val_loss / len(dataloader), metrics


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

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
    scheduler: LRScheduler | None = None,
    config: dict | None = None,
    mode: str = "max",
    threshold: float = 0.5,
    ckpt_cfg: CheckpointConfig | None = None,
    resume_state: dict | None = None,
) -> dict:
    """
    Full training loop with checkpointing and optional resume.

    Checkpointing
    -------------
    Controlled entirely via CheckpointConfig:
      save_best      → best/   (inference, model weights only)
      save_last      → last/   (resumable, full accelerator state)
      every_n_steps  → step-N/ (resumable, full accelerator state)

    Resuming
    --------
    Pass the dict returned by load_checkpoint() as resume_state.
    The loop will start from the correct epoch and skip already-seen batches
    within a partially-completed epoch.

    History / results
    -----------------
    Each epoch appends a record to {save_dir}/train_history.json.
    Best metrics are written to {save_dir}/results.json at the end.
    """
    ckpt_cfg = ckpt_cfg or CheckpointConfig()

    # --- Restore or initialise loop state ---
    start_epoch     = resume_state["epoch"]           if resume_state else 0
    global_step     = resume_state["global_step"]     if resume_state else 0
    best_val_metric = resume_state["best_val_metric"] if resume_state else (
        float("-inf") if mode == "max" else float("inf")
    )

    best_val_metrics_dict: dict = {}

    try:
        for epoch in range(start_epoch, epochs):

            # Pass resume_state only for the first resumed epoch so that
            # mid-epoch batch-skipping is applied exactly once.
            step_resume = resume_state if (resume_state and epoch == start_epoch) else None

            # ---- Training ----
            train_loss, global_step = train_step(
                model, train_dataloader, loss_fn, optimizer,
                accelerator, epoch, epochs,
                scheduler=scheduler,
                ckpt_cfg=ckpt_cfg,
                global_step=global_step,
                resume_state=step_resume,
                config=config,
                best_val_metric=best_val_metric,
            )

            # ---- Validation ----
            val_loss, metrics = val_step(
                model, val_dataloader, loss_fn, accelerator,
                metric_fn=metric_fn,
                threshold=threshold,
            )

            current_metrics    = {"loss": val_loss, **metrics}
            fallback           = float("inf") if mode == "min" else float("-inf")
            current_metric_val = current_metrics.get(primary_metric, fallback)

            # ---- Print ----
            metrics_str = " | ".join(
                f"{CYAN}val_{k}:{RESET} {GREEN}{v:.4f}{RESET}"
                for k, v in metrics.items()
            )
            accelerator.print(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"{CYAN}train_loss:{RESET} {YELLOW}{train_loss:.4f}{RESET} | "
                f"{CYAN}val_loss:{RESET} {YELLOW}{val_loss:.4f}{RESET}"
                + (f" | {metrics_str}" if metrics_str else "")
            )

            # ---- History ----
            if accelerator.is_main_process and ckpt_cfg.save_dir:
                ckpt_cfg.save_dir.mkdir(parents=True, exist_ok=True)
                _append_epoch_history(
                    ckpt_cfg.save_dir,
                    {
                        "epoch":      epoch + 1,
                        "train_loss": train_loss,
                        "val_loss":   val_loss,
                        **{f"val_{k}": v for k, v in metrics.items()},
                    },
                )

            # ---- Best-model checkpoint ----
            if _is_better(current_metric_val, best_val_metric, mode):
                best_val_metric       = current_metric_val
                best_val_metrics_dict = {"val_loss": val_loss, **metrics}

                if ckpt_cfg.save_best and ckpt_cfg.save_dir:
                    _save_best(
                        accelerator, ckpt_cfg.save_dir, model,
                        metrics=best_val_metrics_dict,
                        config=config,
                    )

            # ---- Last-epoch resumable checkpoint ----
            if ckpt_cfg.save_last and ckpt_cfg.save_dir:
                _save_resumable(
                    accelerator, ckpt_cfg.save_dir,
                    tag="last",
                    resume_state={
                        "epoch":           epoch + 1,
                        "global_step":     global_step,
                        "best_val_metric": best_val_metric,
                    },
                    config=config,
                )

    except KeyboardInterrupt:
        accelerator.print(f"\n{YELLOW}Training interrupted by user.{RESET}")
        sys.exit(0)

    # ---- Results ----
    if accelerator.is_main_process and ckpt_cfg.save_dir and best_val_metrics_dict:
        _write_results(ckpt_cfg.save_dir, best_val_metrics_dict)

    accelerator.print(f"{GREEN}Training complete!{RESET}")
    return {"best_metrics": best_val_metrics_dict}


# ---------------------------------------------------------------------------
# eval_model
# ---------------------------------------------------------------------------

def eval_model(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    accelerator: Accelerator,
    metric_fn,
    threshold: float = 0.5,
) -> dict:
    accelerator.print(f"{CYAN}Evaluating model on test set...{RESET}")

    test_loss, metrics = val_step(
        model, dataloader, loss_fn, accelerator,
        metric_fn=metric_fn,
        threshold=threshold,
    )

    metrics_str = " | ".join(
        f"{CYAN}Test {k}:{RESET} {GREEN}{v:.4f}{RESET}" for k, v in metrics.items()
    )
    accelerator.print(
        f"\n{CYAN}Test Loss:{RESET} {YELLOW}{test_loss:.4f}{RESET} | {metrics_str}\n"
    )

    return {"loss": test_loss, **metrics}
