<img width="1351" height="618" alt="image" src="https://github.com/user-attachments/assets/b98d53c1-66e1-44a0-bcba-34a4698dac3d" />
## Running with Docker

Build the image:
```bash
docker build -t flickr8k-captioning-api .
```

Run the container:
```bash
docker run -p 8001:8000 flickr8k-captioning-api
```

Then visit `http://localhost:8001/docs` to test the API.

### Verified Test Result
Successfully tested via the `/docs` interface with a sample image — returned HTTP 200 with a generated caption and Grad-CAM overlay path.

### Notes on Environment Issues Encountered
- OpenCV required additional system-level graphics libraries (`libgl1`, `libglib2.0-0`, `libsm6`, `libxext6`, `libxrender-dev`) not included in the slim Python base image — added via `apt-get install` in the Dockerfile.
- Used the CPU-only build of PyTorch (via `--index-url https://download.pytorch.org/whl/cpu`) to reduce image size and avoid unnecessary GPU dependencies.
- A separate, minimal `app/requirements.txt` was created (instead of reusing the full project `requirements.txt`) to avoid dependency conflicts from unrelated tools (Jupyter, DVC, MLflow, pytest) not needed inside the container.
- The BLIP model is downloaded fresh inside the container on first run (no local cache is shared with the host), so the first request after `docker run` may take a few minutes.
