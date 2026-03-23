import re


def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    text = re.sub(r"http\S+", "", text)         # remove URLs
    text = re.sub(r"\s+", " ", text).strip()    # normalize whitespace
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


def tokenize_batch(batch, tokenizer):
    """Tokenizes title and abstract as a sentence pair."""
    return tokenizer (
        batch["title"],
        batch["abstract"],
        truncation=True,
        padding=False,
        max_length=512
    )

