"""
model.py
Handles loading of the BLIP model and processor.
"""

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration


def get_device():
    """Return best available device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_blip_model(model_name="Salesforce/blip-image-captioning-base"):
    """Load BLIP processor and model, moved to the correct device."""
    device = get_device()
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
    model.eval()
    return processor, model, device


if __name__ == "__main__":
    processor, model, device = load_blip_model()
    print(f"✅ BLIP model loaded on {device}")