import os
import json
import yaml
import argparse
from collections import Counter
from pathlib import Path

# Suppress HuggingFace hub FutureWarnings & logging noise
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from transformers.utils import logging
logging.set_verbosity_error()

from datasets import load_from_disk, DatasetDict
from transformers import AutoTokenizer

from arxiv_paper_discovery.data import encode_labels, tokenize_batch
from arxiv_paper_discovery.config import BASE_DATA_PATH, PROCESSED_DATA_DIR


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", 
                        type=str,
                        required=True)
    return parser.parse_args()

def main():
    args = parse_args()

    print(f"\nLoading configuration from {args.config}...")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    tokenizer_name = config["model"]["pretrained_name"]
    seed = config["experiment"]["seed"]

    print("1. Loading the Base Dataset...")
    dataset = load_from_disk(BASE_DATA_PATH)

    print("\n2. Indentifying the Top 30 Categories...")
    all_categories = [cat for cat_list in dataset["categories"] for cat in cat_list]
    top_30_counts = Counter(all_categories).most_common(30) # Returns a list of tuples
    top_30_categories = [cat for cat, count in top_30_counts]

    class_to_index = {cat: i for i, cat in enumerate(top_30_categories)}

    print("\n3. Filtering dataset to papers with top 30 categories...")
    def has_top_category(example):
        return any(cat in class_to_index for cat in example["categories"])
    
    dataset = dataset.filter(
        has_top_category,
        num_proc=4
    )

    print("\n4. Encoding Muti-Hot label Encoding..")
    dataset = dataset.map(
        encode_labels,
        batched=True,
        fn_kwargs={"category_to_idx": class_to_index},
        num_proc=4
    )

    print("\n5. Splitting into Train/Val/Test (80/10/10)...")
    train_test = dataset.train_test_split(test_size=0.2, seed=seed, shuffle=True)
    test_val = train_test['test'].train_test_split(test_size=0.5, seed=seed, shuffle=True)

    final_dataset = DatasetDict({
        'train': train_test['train'],
        'val': test_val['train'],
        'test': test_val['test']
    })

    print(f"\n6. Tokenizing with {tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    final_dataset = final_dataset.map(
        tokenize_batch,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=4
    )

    # Drop the original text columns to save disk space
    columns_to_remove = ["id", "title", "abstract", "categories", "authors", "update_date"]
    final_dataset = final_dataset.remove_columns(columns_to_remove)

    # Hide non-tensor columns from PyTorch DataLoader and cast to Pytorch tensors
    final_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "token_type_ids", "labels"]
    )

    print(f"\n7. Saving processed dataset to {PROCESSED_DATA_DIR}...")
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    final_dataset.save_to_disk(PROCESSED_DATA_DIR / "arxiv_classification_dataset")

    with open(PROCESSED_DATA_DIR / "category_to_index.json", 'w') as f:
        json.dump(class_to_index, f)

    print("\nProcessed dataset and category mapping saved successfully.")

if __name__ == "__main__":
    main()



