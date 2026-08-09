"""
demo.py
Gradio demo app: upload an image, see the generated caption and Grad-CAM overlay side-by-side.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import gradio as gr  # noqa: E402
from PIL import Image  # noqa: E402

from model import load_blip_model  # noqa: E402
from inference import generate_caption  # noqa: E402
from xai import generate_gradcam, overlay_heatmap  # noqa: E402

# Model ek hi baar load hoga jab app start ho
processor, model, device = load_blip_model()


def caption_and_explain(image: Image.Image):
    """Takes a PIL image, returns (caption, gradcam_overlay_image)."""
    # Temporarily save the uploaded image (functions expect a file path)
    temp_path = "temp_upload.jpg"
    image.convert("RGB").save(temp_path)

    caption = generate_caption(temp_path, model, processor, device)

    cam, _, raw_image = generate_gradcam(temp_path, model, processor, device)
    overlay = overlay_heatmap(raw_image, cam)
    overlay_image = Image.fromarray(overlay)

    os.remove(temp_path)

    return caption, overlay_image


demo = gr.Interface(
    fn=caption_and_explain,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs=[
        gr.Textbox(label="Generated Caption"),
        gr.Image(label="Grad-CAM Overlay (where the model looked)")
    ],
    title="Flickr8k Image Captioning + Explainability Demo",
    description="Upload an image to see the BLIP-generated caption alongside a Grad-CAM heatmap showing which regions influenced the caption."
)

if __name__ == "__main__":
    demo.launch()