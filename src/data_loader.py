"""
data_loader.py
Handles loading and preprocessing of the Flickr8k dataset (captions + images).
"""

import pandas as pd
import os
import re
from PIL import Image


def load_captions(captions_path):
    """Load raw captions file (pipe-delimited)."""
    df = pd.read_csv(captions_path, sep="|")
    df.columns = df.columns.str.strip()
    return df


def clean_text(text):
    """Lowercase and remove punctuation/numbers from a caption."""
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text):
    """Whitespace-based tokenization."""
    return text.split()


def add_start_end_tokens(tokens):
    """Add <start> and <end> markers to a token list."""
    return ['<start>'] + tokens + ['<end>']


def preprocess_captions(df, caption_col="caption_text"):
    """Full text preprocessing pipeline on a captions dataframe."""
    df['clean_caption'] = df[caption_col].apply(clean_text)
    df['tokens'] = df['clean_caption'].apply(tokenize)
    df['final_tokens'] = df['tokens'].apply(add_start_end_tokens)
    return df


def load_image(image_path):
    """Load a single image and convert to RGB."""
    return Image.open(image_path).convert("RGB")


def get_image_paths(image_folder):
    """Return list of image filenames in a folder."""
    return os.listdir(image_folder)


if __name__ == "__main__":
    df = load_captions("../data/captions.txt")
    df = preprocess_captions(df)
    print(f"Loaded and preprocessed {len(df)} captions.")
    print(df.head())