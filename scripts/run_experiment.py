"""Run a hyperparameter sweep using Optuna via Subprocess Orchestration backed by SQLite."""

import argparse
import subprocess
from pathlib import Path

import optuna
import yaml


def main():
    parser = argparse.ArgumentParser(description="Run Optuna Sweep for SciBERT")
    parser.add_argument("--config", type=Path, default=Path("configs/scibert_classification.yaml"))
    parser.add_argument("--sweep-config", type=Path, default=Path("configs/scibert_sweep.yaml"))
    parser.add_argument("--mlflow-dir", type=Path, default=Path("mlruns"))
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--study-name", type=str, default="scibert-sweep")
    parser.add_argument("--storage", type=str, default="sqlite:///scibert_sweep.db")
    cli_args = parser.parse_args()

    with open(cli_args.config) as f:
        base_cfg = yaml.safe_load(f)
        
    metric_mode = base_cfg.get("training", {}).get("metric_mode", "max")
    direction = "maximize" if metric_mode == "max" else "minimize"

    pruner = optuna.pruners.MedianPruner(n_warmup_steps=1)
    
    # Initialize the SQLite database.
    study = optuna.create_study(
        direction=direction, 
        study_name=cli_args.study_name,
        storage=cli_args.storage,
        pruner=pruner,
        load_if_exists=True
    )
    
    print(f"\n{'='*60}")
    print(f"Starting Sweep: {cli_args.study_name}")
    print(f"Database: {cli_args.storage}")
    print(f"{'='*60}\n")

    for i in range(cli_args.n_trials):
        print(f"\n--- Launching Subprocess for Trial {i+1}/{cli_args.n_trials} ---")
        
        cmd = [
            "accelerate", "launch", "scripts/run_training.py",
            "--config", str(cli_args.config),
            "--sweep-config", str(cli_args.sweep_config),
            "--mlflow-dir", str(cli_args.mlflow_dir),
            "--sweep",
            "--study-name", cli_args.study_name,
            "--storage", cli_args.storage,
            "--no-save"
        ]
        
        try:
            # Subprocess runs the trial. Because the script connects to the DB, 
            # Optuna logs metrics and updates the model natively.
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            # Non-zero exits generally indicate a worker crash, not pruning (pruning exits normally).
            print(f"Worker for trial {i+1} failed with exit code {e.returncode}. Moving to next.")
    
    # Reload study at the end to display metrics safely
    study = optuna.load_study(study_name=cli_args.study_name, storage=cli_args.storage)
    
    print(f"\n{'='*60}")
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    
    print(f"Sweep Finished! Completed: {len(completed_trials)} | Pruned: {len(pruned_trials)}")
    if completed_trials:
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best Score: {study.best_value:.4f}")
        print(f"Best Params: {study.best_trial.params}")
    else:
        print("No trials completed successfully.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()