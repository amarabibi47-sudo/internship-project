"""
main.py
FastAPI service for image captioning with Grad-CAM explainability.
"""

import sys
import os
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import cv2

from model import load_blip_model
from inference import generate_caption
from xai import generate_gradcam, overlay_heatmap

app = FastAPI(title="Flickr8k Captioning API")

 
processor, model, device = load_blip_model()

 
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.get("/")
def read_root():
    return {"message": "Flickr8k Captioning API is running. Use POST /caption to generate a caption."}


@app.post("/caption")
async def caption_image(file: UploadFile = File(...)):
    """
    Accepts an image file, returns generated caption + Grad-CAM heatmap path.
    """
     
    temp_filename = f"{uuid.uuid4().hex}_{file.filename}"
    temp_path = os.path.join(OUTPUT_DIR, temp_filename)

    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)

     
    caption = generate_caption(temp_path, model, processor, device)

     
    cam, _, raw_image = generate_gradcam(temp_path, model, processor, device)
    overlay = overlay_heatmap(raw_image, cam)

    
    gradcam_filename = f"gradcam_{temp_filename}.png"
    gradcam_path = os.path.join(OUTPUT_DIR, gradcam_filename)
    overlay_img = Image.fromarray(overlay)
    overlay_img.save(gradcam_path)

    return JSONResponse({
        "caption": caption,
        "gradcam_overlay_path": gradcam_path
    })