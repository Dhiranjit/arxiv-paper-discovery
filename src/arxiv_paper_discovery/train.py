"""
Trainer-first training utilities for multi-label text classification.
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from datasets import load_from_disk
from functools import partial

from sklearn.metrics import precision_recall_fscore_support
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

from arxiv_paper_discovery.label_taxonomy import LABELS, LABEL_TO_IDX, IDX_TO_LABEL
from arxiv_paper_discovery.utils import set_seed


def compute_metrics(eval_pred: EvalPrediction, threshold: float = 0.5) -> dict[str, float]:
    """Trainer-compatible compute_metrics for multi-label outputs."""
    logits, labels = eval_pred

    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs > threshold).astype(np.int32)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0,
    )
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0,
    )

    return {
        "f1": float(f1),
        "f1_weighted": float(f1_weighted),
        "precision": float(precision),
        "recall": float(recall),
    }


def train(
    config: dict[str, Any],
    dataset_path: Path | str,
    output_dir: Path | str,
    resume_from_checkpoint: Path | str | None = None,
    save_model: bool = True,
) -> dict[str, dict[str, float]]:
    """
    Run training directly from config.

    Required config keys: model.pretrained_name, trainer.*
    Paths are passed explicitly so the same config works across local and Kaggle environments.
    """
    # 1. Config
    pretrained_name = config["model"]["pretrained_name"]
    seed            = int(config.get("experiment", {}).get("seed", 42))
    threshold       = float(config.get("training", {}).get("threshold", 0.5))
    sample_ratio    = config.get("training", {}).get("sample_ratio")
    dropout_p       = config.get("model", {}).get("dropout_p")

    set_seed(seed)

    # 2. Run info
    num_epochs = config.get("trainer", {}).get("num_train_epochs", "?")
    mode = "Resuming from checkpoint" if resume_from_checkpoint else "Starting fresh run"
    print(f"\n{'=' * 60}")
    print(f"  {mode}")
    print(f"  Model:       {pretrained_name}")
    print(f"  Epochs:      {num_epochs}")
    print(f"  Dataset:     {dataset_path}")
    print(f"  Output:      {output_dir}")
    if resume_from_checkpoint:
        print(f"  Checkpoint:  {resume_from_checkpoint}")
    if sample_ratio is not None:
        print(f"  Sample:      {float(sample_ratio) * 100:.0f}%")
    print(f"{'=' * 60}\n")

    # 3. Dataset
    dataset = load_from_disk(str(dataset_path))
    train_ds, val_ds = dataset["train"], dataset["val"]


    if sample_ratio is not None and float(sample_ratio) < 1.0:
        sample_ratio = float(sample_ratio)
        train_ds = train_ds.shuffle(seed=seed).select(range(int(len(train_ds) * sample_ratio)))
        val_ds   = val_ds.shuffle(seed=seed + 1).select(range(int(len(val_ds) * sample_ratio)))
        print(f"Sampled {len(train_ds)} train / {len(val_ds)} val examples ({sample_ratio * 100:.0f}%).")

    # 4. Labels — LABELS is the source of truth
    num_labels  = len(LABELS)
    label2id    = LABEL_TO_IDX
    id2label    = IDX_TO_LABEL

    # 5. Tokenizer + Model
    tokenizer    = AutoTokenizer.from_pretrained(pretrained_name)
    model_config = AutoConfig.from_pretrained(
        pretrained_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        label2id=label2id,
        id2label=id2label,
    )

    if dropout_p is not None:
        model_config.classifier_dropout = float(dropout_p)
        
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_name, config=model_config)

    # 6. Collator 
    data_collator = DataCollatorWithPadding(tokenizer)

    # 7. TrainingArguments
    trainer_kwargs = {**config.get("trainer", {}), "output_dir": str(output_dir)}
    if not trainer_kwargs.get("disable_tqdm", True):
        trainer_kwargs["logging_strategy"] = "no"
    if not save_model:
        trainer_kwargs |= {"save_strategy": "no", "load_best_model_at_end": False}

    # 8. Trainer
    trainer = Trainer(
        model=model,
        args=TrainingArguments(**trainer_kwargs),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=partial(compute_metrics, threshold=threshold),
    )

    # 9. Train
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    # 10. Evaluate
    eval_metrics = trainer.evaluate(eval_dataset=val_ds, metric_key_prefix="eval")
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    # 11. Save
    if save_model and trainer.is_world_process_zero():
        trainer.save_model()
        with open(Path(output_dir) / "run_config.yaml", "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)

    return {"train_metrics": train_result.metrics, "eval_metrics": eval_metrics}
