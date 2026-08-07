"""
xai.py
Explainability utilities: Grad-CAM and occlusion-based (SHAP-style) importance maps.
"""

import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image


def generate_gradcam(image_path, model, processor, device):
    """Generate a Grad-CAM heatmap for the BLIP vision encoder."""
    raw_image = Image.open(image_path).convert("RGB")
    inputs = processor(raw_image, return_tensors="pt").to(device)

    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations["value"] = output

    def backward_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0]

    target_layer = model.vision_model.encoder.layers[-1]
    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_full_backward_hook(backward_hook)

    pixel_values = inputs["pixel_values"]
    pixel_values.requires_grad_()

    vision_outputs = model.vision_model(pixel_values=pixel_values)
    image_embeds = vision_outputs[0]

    generated_ids = model.generate(**inputs, max_new_tokens=20)
    caption = processor.decode(generated_ids[0], skip_special_tokens=True)

    score = image_embeds.mean()
    model.zero_grad()
    score.backward()

    grads = gradients["value"]
    acts = activations["value"]

    weights = grads.mean(dim=1, keepdim=True)
    cam = (weights * acts).sum(dim=-1).squeeze(0)

    cam = cam[1:] if cam.shape[0] % 2 != 0 else cam
    num_patches = cam.shape[0]
    grid_size = int(num_patches**0.5)
    cam = cam[: grid_size * grid_size].reshape(grid_size, grid_size)

    cam = F.relu(cam)
    cam = cam.detach().cpu().numpy()
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    handle_fwd.remove()
    handle_bwd.remove()

    return cam, caption, raw_image


def overlay_heatmap(raw_image, cam, alpha=0.5):
    """Overlay a Grad-CAM heatmap on the original image."""
    img_np = np.array(raw_image.resize((224, 224)))
    cam_resized = cv2.resize(cam, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_np, 1 - alpha, heatmap, alpha, 0)
    return overlay


def compute_patch_importance(image_path, model, processor, device, grid_size=4):
    """Occlusion-based (SHAP-style) patch importance map."""
    raw_image = Image.open(image_path).convert("RGB").resize((224, 224))
    img_array = np.array(raw_image)

    inputs = processor(raw_image, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=20)
    original_caption = processor.decode(out[0], skip_special_tokens=True)

    patch_h, patch_w = 224 // grid_size, 224 // grid_size
    importance_map = np.zeros((grid_size, grid_size))

    for i in range(grid_size):
        for j in range(grid_size):
            occluded = img_array.copy()
            occluded[
                i * patch_h : (i + 1) * patch_h, j * patch_w : (j + 1) * patch_w
            ] = 128
            occluded_img = Image.fromarray(occluded)

            occ_inputs = processor(occluded_img, return_tensors="pt").to(device)
            occ_out = model.generate(**occ_inputs, max_new_tokens=20)
            occluded_caption = processor.decode(occ_out[0], skip_special_tokens=True)

            orig_words = set(original_caption.lower().split())
            occ_words = set(occluded_caption.lower().split())
            difference = len(orig_words.symmetric_difference(occ_words))

            importance_map[i, j] = difference

    return importance_map, original_caption
