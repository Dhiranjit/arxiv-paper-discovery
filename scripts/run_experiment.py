"""
Grid search over hyperparameters defined in the sweep section of a YAML config.

Each combination mutates a copy of config and calls train(config) directly.
No model weights are saved — only sweep_summary.json with params and metrics
for every run. To train a final model, re-run the best config with run_training.py.

Usage:
    python scripts/run_experiment.py \\
        --config configs/scibert.yaml \\
        --dataset-path data/processed/tok_scibert_scivocab_uncased \\
        --output-dir outputs/sweep_01
"""

import argparse
import copy
import itertools
import json
import math
import re
from pathlib import Path
from typing import Any

import yaml

from arxiv_paper_discovery.train import train

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_nested(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Write a value into a nested dict using a dotted key, e.g. 'trainer.learning_rate'."""
    keys   = dotted_key.split(".")
    cursor = cfg
    for key in keys[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[keys[-1]] = value


def _make_run_id(param_combo: dict[str, Any]) -> str:
    """Build a short, readable run-id from the sampled param values."""
    slugify = lambda v: re.sub(r"[^a-zA-Z0-9._-]+", "-", str(v).strip())
    parts = []
    for key, val in param_combo.items():
        short_key = slugify(key.split(".")[-1])  # e.g. "learning-rate" from "trainer.learning_rate"
        parts.append(f"{short_key}-{slugify(val)}")
    return "grid_" + "_".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Grid search for SciBERT classifier")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/scibert.yaml"),
        help="Training config YAML (must contain a 'sweep' section)",
    )
    parser.add_argument("--dataset-path", type=Path, required=True, help="Path to tokenized HF dataset")
    parser.add_argument("--output-dir", type=Path, required=True, help="Base directory for sweep run outputs")
    cli_args = parser.parse_args()

    with open(cli_args.config) as f:
        full_cfg = yaml.safe_load(f)

    sweep_params: dict[str, list] = full_cfg.pop("sweep", {})
    if not sweep_params:
        raise ValueError(
            "No 'sweep' section found in the config. "
            "Add a sweep block or use run_training.py directly for a single run."
        )
    for key, values in sweep_params.items():
        if not isinstance(values, list) or len(values) == 0:
            raise ValueError(
                f"Sweep parameter '{key}' must be a non-empty list, got {type(values).__name__}."
            )

    base_output_dir = cli_args.output_dir
    param_names  = list(sweep_params.keys())
    param_values = list(sweep_params.values())
    total_combos = math.prod(len(values) for values in param_values)

    print(f"\n{'='*60}")
    print(f"Grid search: {total_combos} combinations")
    for name, vals in sweep_params.items():
        print(f"  {name}: {vals}")
    print(f"{'='*60}\n")

    all_results: list[dict] = []

    for i, combo in enumerate(itertools.product(*param_values), 1):
        param_combo = dict(zip(param_names, combo))
        run_id      = _make_run_id(param_combo)

        print(f"\n--- Run {i}/{total_combos}: {run_id} ---")
        for k, v in param_combo.items():
            print(f"    {k} = {v}")

        cfg = copy.deepcopy(full_cfg)
        for key, val in param_combo.items():
            _set_nested(cfg, key, val)

        # Mutate config directly per run.
        cfg.setdefault("trainer", {})
        cfg["trainer"]["run_name"] = run_id
        cfg.setdefault("experiment", {})
        cfg["experiment"]["name"] = run_id

        run_output_dir = base_output_dir / run_id
        try:
            results = train(cfg, dataset_path=cli_args.dataset_path, output_dir=run_output_dir,
                            save_model=False)
            best    = results.get("best_metrics", {})
            all_results.append({"run_id": run_id, "params": param_combo, "best_metrics": best})
        except Exception as e:
            print(f"Run {run_id} failed: {e}")
            all_results.append({"run_id": run_id, "params": param_combo, "error": str(e)})

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("Grid search complete\n")

    trainer_cfg = full_cfg.get("trainer", {})
    target_metric = str(trainer_cfg.get("metric_for_best_model", "eval_f1"))
    if target_metric.startswith("eval_"):
        primary_metric = target_metric[len("eval_") :]
    else:
        primary_metric = target_metric
    metric_mode = "max" if bool(trainer_cfg.get("greater_is_better", True)) else "min"

    scored = [
        r for r in all_results
        if "best_metrics" in r and primary_metric in r["best_metrics"]
    ]
    scored.sort(
        key=lambda r: r["best_metrics"][primary_metric],
        reverse=(metric_mode == "max"),
    )

    for rank, r in enumerate(scored, 1):
        score = r["best_metrics"][primary_metric]
        print(f"  #{rank:>2}  {r['run_id']:<45}  {primary_metric}={score:.4f}")

    if scored:
        best_run = scored[0]
        print(f"\nBest run : {best_run['run_id']}")
        print(f"  params : {best_run['params']}")
        print(f"  metrics: {best_run['best_metrics']}")

    # Persist sweep summary even if runs fail (useful for debugging failed sweeps)
    summary_path = base_output_dir / "sweep_summary.json"
    base_output_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary written → {summary_path}")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
