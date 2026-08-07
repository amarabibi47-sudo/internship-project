FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY app/requirements.txt .

# Torch CPU-only version pehle install karein (chhoti, tez)
RUN pip install --no-cache-dir --default-timeout=200 torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Baaki dependencies install karein
RUN pip install --no-cache-dir --default-timeout=200 --retries 5 -r requirements.txt

COPY src/ ./src/
COPY app/ ./app/
COPY data/processed/ ./data/processed/
COPY data/captions.txt ./data/captions.txt

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]