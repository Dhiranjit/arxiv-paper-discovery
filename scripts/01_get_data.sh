# Downloads arXiv dataset (Kaggle) into data/raw/
# Skips download if arxiv-metadata-oai-snapshot.json already exists


set -euo pipefail

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