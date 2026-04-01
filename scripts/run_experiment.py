"""
Grid search over hyperparameters defined in the sweep section of a YAML config.

Each combination calls train() with save_model=False. Results are saved to
sweep_summary.json. Re-run the best config with run_training.py to train a
final model.

Usage:
python scripts/run_experiment.py \
--config configs/scibert.yaml \
--dataset-path data/processed/tok_scibert_scivocab_uncased \
--output-dir experiments/sweep_01
"""

import argparse
import copy
import itertools
import json
import logging
import math
import os
import warnings
from pathlib import Path
from typing import Any

import yaml
from transformers.utils import logging as hf_logging

from arxiv_paper_discovery.train import train

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)
hf_logging.set_verbosity_error()
logging.getLogger("safetensors").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)


def _set_nested(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Write a value into a nested dict using a dotted key, e.g. 'trainer.learning_rate'."""
    keys = dotted_key.split(".")
    cursor = cfg
    for key in keys[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[keys[-1]] = value


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid search for SciBERT classifier")
    parser.add_argument("--config", type=Path, required=True, help="Training config YAML (must contain a 'sweep' section)")
    parser.add_argument("--dataset-path", type=Path, required=True, help="Path to tokenized HF dataset")
    parser.add_argument("--output-dir", type=Path, required=True, help="Base directory for sweep run outputs")
    args = parser.parse_args()

    with open(args.config) as f:
        full_cfg = yaml.safe_load(f)

    sweep_params: dict[str, list] = full_cfg.pop("sweep", {})
    param_names = list(sweep_params.keys())
    param_values = list(sweep_params.values())
    total_combos = math.prod(len(v) for v in param_values)

    print(f"\nGrid search: {total_combos} combinations over {param_names}\n")

    all_results: list[dict] = []

    for i, combo in enumerate(itertools.product(*param_values), 1):
        param_combo = dict(zip(param_names, combo))
        run_id = "_".join(f"{k.rsplit('.', 1)[-1]}={v}" for k, v in param_combo.items())

        print(f"\n[{i}/{total_combos}] {run_id}")

        cfg = copy.deepcopy(full_cfg)
        for key, val in param_combo.items():
            _set_nested(cfg, key, val)

        results = train(cfg, dataset_path=args.dataset_path, output_dir=args.output_dir / run_id,
                        save_model=False)
        all_results.append({"run_id": run_id, "params": param_combo, "eval_metrics": results["eval_metrics"]})

    # Summary
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    metric = full_cfg.get("trainer", {}).get("metric_for_best_model", "eval_f1").removeprefix("eval_")
    scored = sorted(all_results, key=lambda r: r["eval_metrics"].get(metric, 0), reverse=True)

    print(f"\n{'='*60}")
    print(f"Sweep complete — {len(scored)}/{total_combos} runs scored on '{metric}'")
    for rank, r in enumerate(scored, 1):
        print(f"  #{rank}  {r['run_id']}  {metric}={r['eval_metrics'][metric]:.4f}")
    if scored:
        print(f"\nBest: {scored[0]['run_id']}  params={scored[0]['params']}")
    print(f"Summary: {summary_path}\n{'='*60}\n")


if __name__ == "__main__":
    main()
