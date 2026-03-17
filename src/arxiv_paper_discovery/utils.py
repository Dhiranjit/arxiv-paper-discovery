import torch
import random
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import get_scheduler


def flatten_dict(d, parent_key='', sep='.'):
    """
    Recursively flattens a nested dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)



def metric_fn(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculates multi-label classification metrics.
    Assumes y_true and y_pred are 2D binary arrays of shape (n_samples, n_classes).
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    
    accuracy = accuracy_score(y_true, y_pred)
    
    return {
        "accuracy": float(accuracy),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall)
    }


def set_seed(seed: int = 42) -> None:
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_scheduler(sched_cfg: dict, optimizer, epochs: int, steps_per_epoch: int):
    """Returns the instantiated Hugging Face scheduler."""
    sched_name = sched_cfg.get("name")
    if not sched_name:
        return None

    num_training_steps = epochs * steps_per_epoch
    
    warmup_steps = sched_cfg.get("num_warmup_steps", 0)
    warmup_ratio = sched_cfg.get("warmup_ratio", 0.0)
    if warmup_ratio > 0:
        warmup_steps = int(num_training_steps * warmup_ratio)

    return get_scheduler(
        sched_name,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )