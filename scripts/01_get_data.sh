# Download the arXiv dataset using the Kaggle API if it has not already been downloaded.


#!/usr/bin/env bash
set -e

DATA_DIR="data/raw"
DATA_FILE="$DATA_DIR/arxiv/arxiv-metadata-oai-snapshot.json"

if [ -f "$DATA_FILE" ]; then
    echo "Dataset already exists. Skipping download."
else
    mkdir -p $DATA_DIR

    kaggle datasets download Cornell-University/arxiv \
        -p $DATA_DIR \
        --unzip

    echo "Dataset downloaded."
fi