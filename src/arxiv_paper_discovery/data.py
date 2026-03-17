import re
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding


def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    text = re.sub(r"http\S+", "", text)         # remove URLs
    text = re.sub(r"\s+", " ", text).strip()     # normalize whitespace
    return text


def clean_dataset_text(batch):
    """cleans titles and abstracts independently."""
    titles = [clean_text(t) for t in batch["title"]]
    abstracts = [clean_text(a) for a in batch["abstract"]]

    return {"title": titles, "abstract": abstracts}


def process_categories(batch):
    """Splits the space-separated ArXiv categories into lists."""
    cats = [c.split(" ") for c in batch["categories"]]
    return {"categories": cats}


def encode_labels(batch, category_to_idx):
    """Creates multi-hot encoded label vectors for BCEWithLogitsLoss"""
    num_classes = len(category_to_idx)
    batch_labels = []

    for cat_list in batch["categories"]:
        label_vector = [0.0] * num_classes
        for cat in cat_list:
            if cat in category_to_idx:
                label_vector[category_to_idx[cat]] = 1.0
        batch_labels.append(label_vector)
    
    return {"labels": batch_labels}


def tokenize_batch(batch, tokenizer):
    """Tokenizes title and abstract as a sentence pair."""
    return tokenizer (
        batch["title"],
        batch["abstract"],
        truncation=True,
        padding=False,
        max_length=512
    )

def create_dataloader(dataset, tokenizer, batch_size, shuffle=True):
    """Creates a DataLoader with dynamic padding from a formatted HF dataset."""

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator
    )
    return dataloader
