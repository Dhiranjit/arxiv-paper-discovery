import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_DIR = PROJECT_ROOT / "configs"

RAW_DATA_DIR = PROJECT_ROOT / "data/raw"
BASE_DATA_DIR = PROJECT_ROOT / "data/base"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data/processed"

RAW_DATA_PATH = RAW_DATA_DIR / "arxiv/arxiv-metadata-oai-snapshot.json"
BASE_DATA_PATH = BASE_DATA_DIR / "arxiv_base_dataset"