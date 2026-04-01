import torch
import random
import numpy as np
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import get_scheduler
import pandas as pd




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


def compute_label_coverage(dataset, column: str = "categories"):
    """Compute cumulative paper coverage as labels are added in frequency order.

    Coverage is defined as AT LEAST ONE match: a paper is considered "covered"
    as soon as any one of its categories appears in the accumulated label set.
    A paper with multiple categories is counted the first time any of them is
    encountered — it is never double-counted.

    Args:
        dataset: HuggingFace Dataset containing the specified column.
        column:  Name of the column holding per-paper label lists.

    Returns:
        coverage (list[float]): Cumulative % of papers covered after including
                                each successive label (ordered by frequency, descending).
        sorted_labels (list[str]): Labels in the same frequency-descending order.
    """
    
    label_counter = Counter()

    for labels in tqdm(dataset[column], desc="Counting labels"):
        label_counter.update(labels)

    sorted_labels = [l for l, _ in label_counter.most_common()]
    label_sets = [set(labels) for labels in dataset[column]]

    total = len(label_sets)
    covered_mask = [False] * total
    covered_count = 0

    coverage = []

    for label in tqdm(sorted_labels, desc="Computing coverage"):
        for i, paper_labels in enumerate(label_sets):
            if not covered_mask[i] and label in paper_labels:
                covered_mask[i] = True
                covered_count += 1

        coverage.append(covered_count / total * 100)

    return coverage, sorted_labels