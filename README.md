# Flickr8k Image Captioning with Explainability

An end-to-end image captioning pipeline built on the Flickr8k dataset, using a pretrained BLIP model, with explainability (Grad-CAM), MLOps tracking (MLflow, DVC), CI/CD (GitHub Actions), containerization (Docker), and a public demo.

## Project Overview

This project takes an image as input and generates a natural-language caption describing its content, while also providing a visual explanation (Grad-CAM heatmap) of which image regions most influenced the caption. The system uses BLIP (`Salesforce/blip-image-captioning-base`) in a zero-shot setting as the baseline captioning model.

## Architecture
                ┌─────────────────────┐
                │   Flickr8k Dataset   │
                │ (8,091 images +      │
                │  40k+ captions)      │
                └──────────┬───────────┘
                           │
                ┌──────────▼───────────┐
                │   Preprocessing       │
                │ (data_loader.py)      │
                │ - tokenization        │
                │ - image resize/norm   │
                └──────────┬───────────┘
                           │
                ┌──────────▼───────────┐
                │   BLIP Model          │
                │ (model.py)            │
                │ - vision encoder      │
                │ - text decoder        │
                └──────────┬───────────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │

┌──────────▼──────┐ ┌──────▼───────┐ ┌─────▼──────────┐
│ Caption Output │ │ Grad-CAM │ │ MLflow Tracking │
│ (inference.py) │ │ (xai.py) │ │ (registry/runs) │
└──────────┬──────┘ └──────┬───────┘ └─────────────────┘
│ │
└────────┬───────┘
│
┌──────────▼───────────┐
│ FastAPI Service │
│ (app/main.py) │
│ POST /caption │
└──────────┬───────────┘
│
┌──────────▼───────────┐
│ Docker Container │
└──────────┬───────────┘
│
┌──────────▼───────────┐
│ Public Demo │
│ (ngrok tunnel / │
│ Gradio share link) │
└───────────────────────┘

CI/CD: GitHub Actions runs lint (flake8/black), pytest,
and Docker build on every push to main.


## Repository Structure

internship-project/
│
├── data/
│ ├── Images/ # Raw Flickr8k images (not committed, see setup)
│ ├── captions.txt # Raw captions file
│ ├── processed/ # Cleaned captions, vocab, resized images (DVC-tracked)
│ └── xai_examples/ # Curated examples with Grad-CAM/SHAP outputs
│
├── notebooks/ # Exploration, preprocessing, evaluation, XAI notebooks
│
├── src/
│ ├── data_loader.py # Data loading + text preprocessing
│ ├── model.py # BLIP model loading
│ ├── inference.py # Caption generation
│ └── xai.py # Grad-CAM + occlusion-based explainability
│
├── app/
│ ├── main.py # FastAPI service (POST /caption)
│ ├── demo.py # Gradio demo app
│ ├── requirements.txt # Minimal dependencies for the app/container
│ └── README.md # App-specific documentation
│
├── tests/
│ ├── test_pipeline.py # Pytest unit tests
│ └── sample_data/ # Sample image for CI testing
│
├── .github/workflows/ci.yml # CI/CD pipeline (lint, test, Docker build)
├── Dockerfile
├── .dockerignore
├── .gitignore
├── setup.cfg # Flake8 configuration
├── requirements.txt # Full environment dependencies (local dev)
└── README.md # This file


## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/amarabibi47-sudo/internship-project.git
cd internship-project
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the Flickr8k dataset
Download from [Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k) and place:
- Images into `data/Images/`
- Captions file as `data/captions.txt` (pipe-delimited: `image_name|caption_number|caption_text`)

### 5. Run the notebooks (in order)
1. `01_eda_flickr8k.ipynb` - data exploration
2. `03_preprocessing.ipynb` - text/image preprocessing
3. `04_blip_zero_shot.ipynb` - zero-shot BLIP evaluation
4. `05_baseline_evaluation.ipynb` - BLEU/METEOR/ROUGE evaluation
5. `09_xai_example_selection.ipynb` through `12_xai_comparison_mlflow.ipynb` - explainability

### 6. Run the FastAPI service locally
```bash
uvicorn app.main:app --reload
```
Visit `http://127.0.0.1:8000/docs` to test the `/caption` endpoint.

### 7. Run the Gradio demo
```bash
python app/demo.py
```

### 8. Run with Docker
```bash
docker build -t flickr8k-captioning-api .
docker run -p 8000:8000 flickr8k-captioning-api
```

### 9. Run tests and linting
```bash
black --check src/ app/ tests/
flake8 src/ app/ tests/
pytest tests/ -v
```

## Model

- **Base model:** `Salesforce/blip-image-captioning-base` (Hugging Face Transformers)
- **Usage mode:** Zero-shot (no fine-tuning performed on Flickr8k)
- **Decoding strategy:** Greedy decoding (selected as baseline after comparing against beam search, see Week 2 Day 4 results)
- **Registered in:** MLflow Model Registry as `flickr8k-blip-baseline`, version 1

## Evaluation Results (Zero-Shot Baseline, 200-image validation subset)

| Metric | Score |
|---|---|
| BLEU | 0.2025 |
| METEOR | 0.3882 |
| ROUGE-L | 0.5277 |

## Explainability

- **Grad-CAM:** Applied to the BLIP vision encoder's last layer to visualize which image regions most influenced the generated caption.
- **Occlusion-based (SHAP-style) importance:** Patches of the image were masked one at a time to measure their effect on the generated caption.
- **Finding:** Poor-scoring captions often occurred when the model prioritized a visually dominant background/texture (e.g., rock, water, sky) over a smaller, less visually prominent subject (e.g., people, small animals).

## CI/CD

GitHub Actions (`.github/workflows/ci.yml`) automatically runs on every push/PR to `main`:
1. **Lint & format check** - `black --check` and `flake8`
2. **Unit tests** - `pytest` on `data_loader.py` and `inference.py`
3. **Docker build** - builds the container image (runs only on push to `main`, after lint/test pass)

## Deployment

The FastAPI service was containerized with Docker and exposed publicly for testing via an ngrok tunnel, since free-tier cloud platforms (Hugging Face Spaces, Render) now require card verification even for their no-cost tiers. The container was tested end-to-end with the public tunnel using both Flickr8k images and new, out-of-dataset images to confirm generalization.

## Known Limitations

- **No fine-tuning performed:** The model is used zero-shot; captions reflect general pretraining knowledge rather than Flickr8k-specific caption style.
- **CPU-only inference:** No GPU was available in this environment, so inference and Grad-CAM generation are slower than a GPU-based deployment would be.
- **Background-dominance bias:** The model sometimes describes a visually dominant background/texture instead of a smaller foreground subject (see Explainability section).
- **Deployment is not fully persistent:** The public demo link (ngrok) requires the local machine to remain running; a permanent cloud deployment was not completed due to free-tier card-verification requirements on all evaluated platforms (Hugging Face Spaces, Render, Azure).
- **Grad-CAM resolution:** Heatmaps are limited by the vision transformer's patch grid resolution, giving a coarser localization than pixel-level methods.

## Future Work

- Fine-tune BLIP's text decoder (with the vision encoder frozen, as explored in Week 2 Day 3) on the full Flickr8k training set to improve caption style alignment and BLEU/METEOR/ROUGE scores.
- Complete a persistent cloud deployment (Azure Container Apps, Render, or Hugging Face Spaces) once card verification is completed.
- Export the model to ONNX format for faster, hardware-agnostic inference (see `notebooks/14_onnx_export.ipynb` for an initial exploration).
- Set up data/prediction drift monitoring (e.g., Evidently AI) for production readiness.
- Expand the evaluation set beyond 200 images to the full validation split for more statistically robust metrics.

## Author

Amara Bibi

## Mentor

Mateen Yaqoob