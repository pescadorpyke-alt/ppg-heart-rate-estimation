# Hugging Face Space (Docker SDK) image for the PPG-HR inference API.
# HF builds this at the Space root and serves the app on port 7860.
FROM python:3.12-slim

WORKDIR /app

# CPU-only PyTorch from the dedicated index (small image, no CUDA).
RUN pip install --no-cache-dir torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY api/ ./api/
COPY best.pt ./best.pt

# Model is baked into the image (no S3 needed on HF).
ENV MODEL_LOCAL_PATH=/app/best.pt

EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
