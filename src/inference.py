"""
inference.py
Handles caption generation (inference) using a loaded BLIP model.
"""

from PIL import Image


def generate_caption(image_path, model, processor, device, max_new_tokens=30, num_beams=1):
    """
    Generate a caption for a single image.
    num_beams=1 -> greedy decoding (chosen baseline from Week 2)
    """
    raw_image = Image.open(image_path).convert("RGB")
    inputs = processor(raw_image, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=num_beams)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption


def generate_captions_batch(image_paths, model, processor, device):
    """Generate captions for a list of image paths."""
    results = []
    for path in image_paths:
        caption = generate_caption(path, model, processor, device)
        results.append({"image": path, "caption": caption})
    return results


if __name__ == "__main__":
    from model import load_blip_model
    processor, model, device = load_blip_model()
    caption = generate_caption("../data/Images/1000268201_693b08cb0e.jpg", model, processor, device)
    print(f"Generated caption: {caption}")