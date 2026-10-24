# Hugging Face Space (Docker SDK) image for the PPG-HR inference API.
# HF builds this at the Space root and serves the app on port 7860.
# Serves the committed edge model (models/ppg_hr.onnx) with ONNX Runtime: no
# PyTorch in the image (DECISIONS.md D23).
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY api/ ./api/
COPY models/ppg_hr.onnx models/ppg_hr.json ./models/

EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
