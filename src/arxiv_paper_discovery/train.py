"""
Trainer-first training utilities for multi-label text classification.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from arxiv_paper_discovery.label_taxonomy import GROUPS
from arxiv_paper_discovery.utils import set_seed


def read_model_threshold(model, override: float | None = None) -> float:
    """Read the classification threshold from model config, with optional CLI override."""
    if override is not None:
        return float(override)
    task_params = getattr(model.config, "task_specific_params", {}) or {}
    return float(task_params.get("threshold", 0.5))


def build_compute_metrics(threshold: float):
    """Return a Trainer-compatible compute_metrics callable for multi-label outputs."""

    def compute_metrics(eval_pred) -> dict[str, float]:
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]

        probs = 1.0 / (1.0 + np.exp(-logits))
        preds = (probs > threshold).astype(np.int32)
        labels = labels.astype(np.int32)

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            preds,
            average="macro",
            zero_division=0,
        )
        accuracy = accuracy_score(labels, preds)

        return {
            "accuracy": float(accuracy),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
        }

    return compute_metrics


def train(
    config: dict[str, Any],
    dataset_path: Path | str,
    output_dir: Path | str,
) -> dict[str, dict[str, float]]:
    """
    Run training directly from config.

    Required config sections:
    - model.pretrained_name
    - trainer (TrainingArguments kwargs, passed directly)

    Paths are passed explicitly rather than read from config so the same
    config file works unchanged across local and Kaggle/Colab environments.
    """

    experiment_cfg = config.get("experiment") or {}
    model_cfg = config.get("model") or {}
    training_cfg = config.get("training") or {}

    pretrained_name = model_cfg.get("pretrained_name")
    if not pretrained_name:
        raise ValueError("Missing required config: model.pretrained_name")

    seed = int(experiment_cfg.get("seed", 42))
    set_seed(seed)

    dataset_path = Path(dataset_path)

    trainer_kwargs = dict(config.get("trainer") or {})
    trainer_kwargs["output_dir"] = str(output_dir)
    trainer_kwargs.setdefault("seed", seed)
    trainer_kwargs.setdefault("data_seed", seed)
    trainer_kwargs.setdefault("label_names", ["labels"])
    trainer_kwargs.setdefault("remove_unused_columns", True)

    threshold = float(training_cfg.get("threshold", 0.5))
    resume_from_checkpoint = training_cfg.get("resume_from_checkpoint")

    dataset = load_from_disk(str(dataset_path))
    train_dataset = dataset["train"]
    val_dataset = dataset["val"]

    sample_ratio = training_cfg.get("sample_ratio")
    if sample_ratio is not None:
        sample_ratio = float(sample_ratio)
        if not (0.0 < sample_ratio <= 1.0):
            raise ValueError("training.sample_ratio must be in the range (0, 1].")
        if sample_ratio < 1.0:
            train_limit = max(1, int(len(train_dataset) * sample_ratio))
            val_limit = max(1, int(len(val_dataset) * sample_ratio))
            train_dataset = train_dataset.shuffle(seed=seed).select(range(train_limit))
            val_dataset = val_dataset.shuffle(seed=seed + 1).select(range(val_limit))
            print(
                f"Sampled {train_limit} train / {val_limit} val examples "
                f"({sample_ratio * 100:.0f}%)."
            )

    if len(train_dataset) == 0:
        raise ValueError("Train split is empty.")

    first_labels = train_dataset[0].get("labels")
    if not isinstance(first_labels, list) or len(first_labels) == 0:
        raise ValueError("Expected train dataset items to contain non-empty list field: labels")

    num_labels = len(first_labels)
    if len(GROUPS) != num_labels:
        raise ValueError(
            f"Dataset label width ({num_labels}) does not match taxonomy size ({len(GROUPS)})."
        )
    label_names = list(GROUPS)

    label2id = {name: idx for idx, name in enumerate(label_names)}
    id2label = {idx: name for idx, name in enumerate(label_names)}

    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model_config = AutoConfig.from_pretrained(
        pretrained_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        label2id=label2id,
        id2label=id2label,
    )
    if model_cfg.get("dropout_p") is not None:
        model_config.classifier_dropout = float(model_cfg["dropout_p"])
    task_specific_params = dict(getattr(model_config, "task_specific_params", {}) or {})
    task_specific_params["threshold"] = threshold
    model_config.task_specific_params = task_specific_params

    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_name,
        config=model_config,
    )

    training_args = TrainingArguments(**trainer_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(threshold),
    )

    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    train_metrics = train_result.metrics
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    trainer.save_state()

    eval_metrics = trainer.evaluate(eval_dataset=val_dataset, metric_key_prefix="eval")
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    if trainer.is_world_process_zero():
        trainer.save_model()
        tokenizer.save_pretrained(training_args.output_dir)

        output_dir = Path(training_args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "run_config.yaml", "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)

    return {
        "train_metrics": train_metrics,
        "best_metrics": eval_metrics,
    }
