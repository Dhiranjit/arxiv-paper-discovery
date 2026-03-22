"""
src/arxiv_paper_discovery/predictor.py

Persistent inference engine for multi-label arxiv paper tagging.
Loads model weights once and exposes a fast .predict() method.

Checkpoint layout expected (saved_models/<run-id>/best/):
    model.pth      ← state dict only (from _save_best in train.py)
    config.yaml    ← training config bundled alongside the weights
    metrics.json   ← best val metrics (informational)

Multi-label note
----------------
Training uses BCEWithLogitsLoss, so each class is an independent binary
decision.  We apply sigmoid + threshold (read from the config) rather than
softmax + argmax.  A single article can therefore be assigned zero or more
tags, and the result carries per-class probabilities for every predicted tag.
"""

import json
from pathlib import Path

import torch
import yaml
from transformers import BertModel, BertTokenizer

from arxiv_paper_discovery.config import PROCESSED_DATA_DIR
from arxiv_paper_discovery.data import clean_text
from arxiv_paper_discovery.models import SciBERTClassifier

CYAN  = "\033[96m"
RESET = "\033[0m"


class ArticleTagger:
    """
    Persistent inference engine.  Load once, call .predict() many times.

    Parameters
    ----------
    checkpoint_dir : Path
        Directory produced by _save_best() in train.py, containing
        model.pth and config.yaml (e.g. saved_models/<run-id>/best/).
    device : str | None
        Force a device ('cpu', 'cuda', 'mps').  Auto-detected when None.
    threshold : float | None
        Override the classification threshold.  When None the value from
        config.yaml (training.threshold) is used, falling back to 0.5.
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        device: str | None = None,
        threshold: float | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"{CYAN}Inference engine initialising on: {self.device.upper()}{RESET}")

        checkpoint_dir = Path(checkpoint_dir)
        model_path  = checkpoint_dir / "model.pth"
        config_path = checkpoint_dir / "config.yaml"

        if not model_path.exists():
            raise FileNotFoundError(
                f"model.pth not found in '{checkpoint_dir}'. "
                "Point --checkpoint at a best/ directory produced by training."
            )
        if not config_path.exists():
            raise FileNotFoundError(
                f"config.yaml not found in '{checkpoint_dir}'. "
                "The best/ checkpoint should contain a bundled config."
            )

        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        model_cfg      = self.cfg["model"]
        train_cfg      = self.cfg.get("training", {})
        self.threshold = threshold if threshold is not None else train_cfg.get("threshold", 0.5)

        print(f"{CYAN}Loading weights from : {model_path}{RESET}")
        print(f"{CYAN}Classification threshold : {self.threshold}{RESET}")

        # Label mapping: index → class name
        with open(PROCESSED_DATA_DIR / "group_to_index.json") as f:
            group_to_index = json.load(f)
        self.index_to_class: dict[int, str] = {v: k for k, v in group_to_index.items()}

        # Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_cfg["pretrained_name"])

        # Model
        scibert = BertModel.from_pretrained(model_cfg["pretrained_name"])
        self.model = SciBERTClassifier(
            llm=scibert,
            dropout_p=model_cfg["dropout_p"],
            num_classes=len(self.index_to_class),
        ).to(self.device)

        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"{CYAN}Model ready.{RESET}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def predict(
        self,
        title:       str | list[str],
        abstract:    str | list[str] = "",
    ) -> dict | list[dict]:
        """
        Tag one or more arxiv papers.

        Parameters
        ----------
        title    : single string or list of strings
        abstract : single string, list of strings, or "" for all

        Returns
        -------
        Single dict when title is a str, list of dicts otherwise.
        Each dict contains:
            tags          : list[str]   — predicted class names (may be empty)
            probabilities : dict[str, float] — sigmoid score for every predicted tag
            text          : str         — cleaned input text fed to the model
        """
        is_single = isinstance(title, str)

        titles    = [title]    if is_single else list(title)
        abstracts = [abstract] if is_single else (
            [abstract] * len(titles) if isinstance(abstract, str) else list(abstract)
        )

        raw_texts     = [f"{t} {a}".strip() for t, a in zip(titles, abstracts)]
        cleaned_texts = [clean_text(t) for t in raw_texts]

        inputs = self.tokenizer(
            cleaned_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        logits = self.model(inputs)                          # (B, num_classes)
        probs  = torch.sigmoid(logits)                       # independent per class
        preds  = (probs > self.threshold).int()              # (B, num_classes)

        results = []
        for i in range(len(cleaned_texts)):
            active_indices = preds[i].nonzero(as_tuple=True)[0].tolist()
            tags           = [self.index_to_class[idx] for idx in active_indices]
            tag_probs      = {self.index_to_class[idx]: round(probs[i, idx].item(), 4)
                              for idx in active_indices}
            results.append({
                "tags":          tags,
                "probabilities": tag_probs,
                "text":          cleaned_texts[i],
            })

        return results[0] if is_single else results         