"""
Persistent inference engine for multi-label arXiv paper tagging.

Supported model format:
- Hugging Face save_pretrained() directory containing model + tokenizer files.
- If thresholds.json is present alongside the model, per-class thresholds are used;
  otherwise falls back to a scalar threshold of 0.5.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from arxiv_paper_discovery.data import clean_text

CYAN = "\033[96m"
RESET = "\033[0m"


def _read_index_to_class(model) -> dict[int, str]:
    raw_id2label = getattr(model.config, "id2label", None) or {}
    index_to_class: dict[int, str] = {}

    for raw_idx, raw_name in raw_id2label.items():
        try:
            idx = int(raw_idx)
        except (TypeError, ValueError):
            continue
        index_to_class[idx] = str(raw_name)

    if index_to_class:
        return index_to_class

    num_labels = int(getattr(model.config, "num_labels", 0) or 0)
    return {i: f"LABEL_{i}" for i in range(num_labels)}


def _load_thresholds(
    checkpoint_dir: Path, index_to_class: dict[int, str], device: str
) -> torch.Tensor | float:
    thresholds_path = checkpoint_dir / "thresholds.json"
    if not thresholds_path.exists():
        return 0.5
    thresholds_dict = json.loads(thresholds_path.read_text())
    values = [thresholds_dict.get(index_to_class[i], 0.5) for i in range(len(index_to_class))]
    return torch.tensor(values, dtype=torch.float32, device=device)


class ArticleTagger:
    """
    Persistent inference engine. Load once, call .predict() many times.
    """

    def __init__(self, checkpoint_dir: Path, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"{CYAN}Inference engine initialising on: {self.device.upper()}{RESET}")

        checkpoint_dir = Path(checkpoint_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir).to(self.device)
        self.model.eval()

        self.index_to_class = _read_index_to_class(self.model)
        self.thresholds = _load_thresholds(checkpoint_dir, self.index_to_class, self.device)

        threshold_info = (
            "per-class (thresholds.json)"
            if isinstance(self.thresholds, torch.Tensor)
            else f"scalar {self.thresholds}"
        )
        print(f"{CYAN}Loaded HF model from: {checkpoint_dir}{RESET}")
        print(f"{CYAN}Thresholds           : {threshold_info}{RESET}")
        print(f"{CYAN}Model ready.{RESET}")

    @torch.inference_mode()
    def predict(
        self,
        title: str | list[str],
        abstract: str | list[str] = "",
    ) -> dict | list[dict]:
        """
        Tag one or more arXiv papers.
        """
        is_single = isinstance(title, str)

        titles = [title] if is_single else list(title)
        abstracts = [abstract] if is_single else (
            [abstract] * len(titles) if isinstance(abstract, str) else list(abstract)
        )

        cleaned_titles = [clean_text(t) for t in titles]
        cleaned_abstracts = [clean_text(a) for a in abstracts]

        inputs = self.tokenizer(
            cleaned_titles,
            cleaned_abstracts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        logits = self.model(**inputs).logits
        probs = torch.sigmoid(logits)
        preds = (probs > self.thresholds).int()

        results = []
        for i in range(len(titles)):
            active_indices = preds[i].nonzero(as_tuple=True)[0].tolist()
            tags = [self.index_to_class.get(idx, f"LABEL_{idx}") for idx in active_indices]
            tag_probs = {
                self.index_to_class.get(idx, f"LABEL_{idx}"): round(probs[i, idx].item(), 4)
                for idx in active_indices
            }
            results.append({"tags": tags, "probabilities": tag_probs})

        return results[0] if is_single else results
