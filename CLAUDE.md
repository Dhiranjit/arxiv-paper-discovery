# CLAUDE.md — arxiv-paper-discovery

> **Work in progress.** The codebase is currently in an active, messy state.
> Expect incomplete parts, temporary workarounds, and ongoing restructuring.

## Project Overview

SciBERT-based multi-label classifier that tags arXiv papers into a curated research taxonomy.
The taxonomy contains **23 grouped domains** spanning computer science, physics, mathematics,
and quantitative research areas.

**Current focus:** Robust paper classification and inference.
**Planned:** Semantic search and paper recommendation.

Runs in two environments:

- **Local machine** — paths passed via CLI, default batch/precision settings in config
- **Kaggle / Google Colab** — same config, adjust `fp16` and `per_device_train_batch_size` for GPU

Logging is **notebook-friendly and toggleable** — `tqdm` renders poorly in notebook commit runs.
Dependencies and packaging are managed through `pyproject.toml`.

---

## Core Stack

| Purpose                      | Library                                  |
| ---------------------------- | ---------------------------------------- | --- |
| Model execution and training | PyTorch                                  |
| Model + tokenizer            | HuggingFace Transformers (SciBERT)       | 2   |
| Datasets                     | HuggingFace Datasets (disk-backed Arrow) |
| Training                     | HuggingFace Trainer + Accelerate         |
| Evaluation                   | scikit-learn (multi-label metrics)       |
| Config management            | YAML                                     |
| Packaging                    | `pyproject.toml`                         |

---

## Directory Structure

```
configs/
    scibert.yaml                      # single config for all environments (paths passed via CLI)

data/
    raw/arxiv/                        # raw arXiv snapshot JSON — READ ONLY, never modify
    base/arxiv_base_dataset/          # cleaned HuggingFace dataset (minimal columns)
    processed/arxiv_taxonomy_dataset/ # dataset with grouped taxonomy labels
    processed/group_to_index.json     # taxonomy class-to-index mapping
    processed/tok_scibert_scivocab_uncased/  # tokenized dataset ready for training

scripts/
    01_get_data.sh                    # download raw arXiv metadata from Kaggle
    02_create_base_dataset.py         # clean dataset + select required columns
    03_build_taxonomy_dataset.py      # map labels to grouped taxonomy + multi-hot encoding
    04_tokenize_dataset.py            # tokenize title+abstract and save Arrow dataset
    run_training.py                   # single training run from YAML config
    run_experiment.py                 # grid search over config sweep parameters
    run_eval.py                       # evaluate trained model on test split
    run_inference.py                  # offline batch inference on JSONL or HF dataset
    run_serve.py                      # FastAPI + Ray Serve online inference service
    sample_dataset.py                 # create small dataset samples for experiments/uploads

src/arxiv_paper_discovery/
    config.py                         # project-wide constants and path helpers
    data.py                           # text cleaning, dataset utilities, tokenization helpers
    label_taxonomy.py                 # arXiv category → grouped taxonomy mapping + multi-hot encoding
    train.py                          # Trainer pipeline, metrics, model setup, artifact saving
    predictor.py                      # persistent inference engine (ArticleTagger)
    utils.py                          # reproducibility utilities and helper functions
    web/dashboard.html                # simple UI for the inference service
```

---

## Config-Driven Workflow

Configs are the **primary source of truth** for experiment settings — model name, training
parameters, classification thresholds, and hyperparameter sweeps.

The exception is **environment-specific paths**: these are always passed via CLI so the same
config works across local and Kaggle/Colab environments without modification.

---

## Path Handling

For execution scripts (`run_training.py`, `run_experiment.py`), the following are **always
provided as CLI arguments**, never stored in config:

- `--dataset-path` — path to the tokenized HuggingFace dataset
- `--output-dir` — directory for saved model and artifacts
- `--resume-dir` — checkpoint directory to resume from (optional; `run_training.py` only)

---

## Training Execution

Training scripts should be launched with **Accelerate**.

Example (local):

```bash
accelerate launch scripts/run_training.py \
    --config configs/scibert.yaml \
    --dataset-path data/processed/tok_scibert_scivocab_uncased \
    --output-dir outputs/run_01
```

---

## Design Philosophy

Keep the system simple, explicit, and readable.

- Prefer flat, readable code over clever abstractions
- Avoid unnecessary wrappers around HuggingFace Trainer
- Do not introduce complex logic where a small change would suffice
- If something requires too many workarounds to fit together, revisit the design instead

---

## Updating This File

Claude may update this document when:

- Directory structures change or how code works changes
- New scripts are introduced
- Commands or paths change

However:

- Do not modify the Design Philosophy
- Only extend or clarify documentation where necessary
