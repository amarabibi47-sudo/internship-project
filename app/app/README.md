# Flickr8k Captioning API

A FastAPI service that accepts an image and returns a generated caption along with a Grad-CAM explainability heatmap overlay, built on top of the fine-tuned/zero-shot BLIP model from Weeks 2-3.

## Endpoints

### `GET /`
Health check endpoint — confirms the API is running.

### `POST /caption`
Accepts an image file upload and returns:
- `caption`: the generated caption text (from BLIP, greedy decoding)
- `gradcam_overlay_path`: file path to the saved Grad-CAM heatmap overlay image, showing which image regions most influenced the caption

 

## Testing

The API was tested locally in two ways:
1. **FastAPI's built-in `/docs` interface** — uploading a sample image directly through the browser
2. **curl command** — sending a POST request with an image file from the terminal

Both methods returned a valid caption and a saved Grad-CAM overlay path.

 

 <img width="1366" height="126" alt="image" src="https://github.com/user-attachments/assets/361c1d0d-be94-4997-9104-919ce34fda09" />
<img width="1296" height="548" alt="image" src="https://github.com/user-attachments/assets/d919af83-102d-487a-80a8-0de0f88f601c" />
<img width="1300" height="523" alt="image" src="https://github.com/user-attachments/assets/60d3e4bf-5a17-492b-857c-ecd279b2105b" />
