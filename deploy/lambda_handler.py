"""AWS Lambda entrypoint.

Wraps the existing FastAPI app with Mangum so the *same* API code runs on Lambda
behind a Function URL / API Gateway — no rewrite, no behaviour drift vs. local.

The model is loaded from S3 on the first invocation (cold start). Required Lambda
env vars:
    MODEL_S3_BUCKET, MODEL_S3_KEY   -> where the ONNX model lives (optional: the
                                       image also contains models/ppg_hr.onnx)
    MODEL_LOCAL_PATH=/tmp/ppg_hr.onnx -> only /tmp is writable on Lambda
"""
from mangum import Mangum

from api.main import app

handler = Mangum(app, lifespan="off")
