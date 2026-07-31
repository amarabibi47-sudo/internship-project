"""
test_pipeline.py
Unit tests for data_loader.py and inference.py
"""

import sys
import os
import pandas as pd
from PIL import Image

# src/ folder ko path mein add karein taake imports ho sakein
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import clean_text, tokenize, add_start_end_tokens, load_captions
from model import load_blip_model
from inference import generate_caption


# ---------------------------
# data_loader.py ke tests
# ---------------------------

def test_clean_text_lowercases_and_removes_punctuation():
    result = clean_text("A Dog is Running!!")
    assert result == "a dog is running"


def test_clean_text_handles_empty_string():
    result = clean_text("")
    assert result == ""


def test_clean_text_handles_numbers():
    result = clean_text("There are 5 dogs here")
    assert "5" not in result


def test_tokenize_returns_list():
    tokens = tokenize("a dog is running")
    assert isinstance(tokens, list)
    assert len(tokens) == 4


def test_tokenize_empty_string():
    tokens = tokenize("")
    assert tokens == []


def test_add_start_end_tokens():
    tokens = ["a", "dog"]
    result = add_start_end_tokens(tokens)
    assert result[0] == "<start>"
    assert result[-1] == "<end>"
    assert len(result) == 4


def test_load_captions_returns_dataframe():
    df = load_captions("data/captions.txt")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "image_name" in df.columns
    assert "caption_text" in df.columns


# ---------------------------
# inference.py ke tests
# ---------------------------

def test_generate_caption_returns_string():
    processor, model, device = load_blip_model()
    sample_image = "data/Images/1000268201_693b08cb0e.jpg"

    caption = generate_caption(sample_image, model, processor, device)

    assert isinstance(caption, str)
    assert len(caption) > 0


def test_generate_caption_no_crash_on_valid_image():
    processor, model, device = load_blip_model()
    sample_image = "data/Images/1000268201_693b08cb0e.jpg"

    try:
        caption = generate_caption(sample_image, model, processor, device)
        success = True
    except Exception:
        success = False

    assert success is True


def test_generate_caption_invalid_path_raises_error():
    processor, model, device = load_blip_model()
    invalid_path = "data/Images/this_image_does_not_exist.jpg"

    try:
        generate_caption(invalid_path, model, processor, device)
        raised_error = False
    except Exception:
        raised_error = True

    assert raised_error is True