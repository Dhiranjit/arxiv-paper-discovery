"""
Grid search over hyperparameters defined in the sweep section of a YAML config.

Usage
-----
    python scripts/run_experiment.py --config configs/scibert_classification.yaml

Sweep config format (added to the same YAML or a separate file)
---------------------------------------------------------------
sweep:
  optimizer.lr:     [1e-5, 3e-5, 4e-5, 1e-4]
  model.dropout_p:  [0.1, 0.3, 0.5]

All combinations are run in sequence.  Results are collected in memory and a
summary table is printed at the end.  Each run writes its own train_history.json
and results.json under saved_models/<run-id>/.
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_nested(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Write a value into a nested dict using a dotted key, e.g. 'optimizer.lr'."""
    keys   = dotted_key.split(".")
    cursor = cfg
    for key in keys[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[keys[-1]] = value


def _slugify(value: Any) -> str:
    """Convert sweep values to filesystem-safe fragments."""
    text = str(value).strip()
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", text)


def _make_run_id(param_combo: dict[str, Any]) -> str:
    """Build a short, readable run-id from the sampled param values."""
    parts = []
    for key, val in param_combo.items():
        short_key = _slugify(key.split(".")[-1])  # e.g. "lr" from "optimizer.lr"
        parts.append(f"{short_key}-{_slugify(val)}")
    return "grid_" + "_".join(parts)


def _validate_sweep(sweep_params: dict[str, Any]) -> None:
    """Fail fast when sweep values are not list-like parameter choices."""
    for key, values in sweep_params.items():
        if not isinstance(values, list) or len(values) == 0:
            raise ValueError(
                f"Sweep parameter '{key}' must be a non-empty list, got {type(values).__name__}."
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Grid search for SciBERT classifier")
    parser.add_argument("--config",   type=Path, default=Path("configs/scibert_classification.yaml"),
                        help="Training config YAML (must contain a 'sweep' section)")
    parser.add_argument("--save-dir", type=Path, default=Path("saved_models"))
    parser.add_argument("--no-save",  action="store_true",
                        help="Disable checkpointing for all runs (dry-run / quick test)")
    parser.add_argument(
        "--tokenized-data-dir",
        type=Path,
        default=None,
        help="Path to tokenized dataset directory (overrides config/default for all runs)",
    )
    parser.add_argument(
        "--label-map-path",
        type=Path,
        default=None,
        help="Path to group_to_index.json (overrides config/default for all runs)",
    )
    cli_args = parser.parse_args()

    with open(cli_args.config) as f:
        full_cfg = yaml.safe_load(f)

    sweep_params: dict[str, list] = full_cfg.pop("sweep", {})
    if not sweep_params:
        raise ValueError(
            "No 'sweep' section found in the config. "
            "Add a sweep block or use run_training.py directly for a single run."
        )
    _validate_sweep(sweep_params)

    param_names  = list(sweep_params.keys())
    param_values = list(sweep_params.values())
    total_combos = math.prod(len(values) for values in param_values)

    # Import lazily so --help and static config checks do not require heavy training deps.
    from run_training import execute_training_run  # reuse the same core function

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

        # Build a minimal args namespace matching what execute_training_run expects
        args = argparse.Namespace(
            config             = cli_args.config,
            run_id             = run_id,
            no_save            = cli_args.no_save,
            resume             = None,
            save_dir           = cli_args.save_dir,
            tokenized_data_dir = cli_args.tokenized_data_dir,
            label_map_path     = cli_args.label_map_path,
        )

        try:
            results = execute_training_run(args, cfg)
            best    = results.get("best_metrics", {})
            all_results.append({"run_id": run_id, "params": param_combo, "best_metrics": best})
        except Exception as e:
            print(f"Run {run_id} failed: {e}")
            all_results.append({"run_id": run_id, "params": param_combo, "error": str(e)})

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("Grid search complete\n")

    primary_metric = full_cfg.get("training", {}).get("primary_metric", "f1")
    metric_mode    = full_cfg.get("training", {}).get("metric_mode", "max")

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
    summary_path = cli_args.save_dir / "sweep_summary.json"
    cli_args.save_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary written → {summary_path}")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
