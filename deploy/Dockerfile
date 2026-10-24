# AWS Lambda container image for the PPG-HR inference API.
# Build context = project root:  docker build -f deploy/Dockerfile -t ppg-hr-api .
#
# Python 3.12 base (AWS Lambda runtime). The API serves models/ppg_hr.onnx with
# ONNX Runtime, so the image needs no PyTorch (DECISIONS.md D23).
FROM public.ecr.aws/lambda/python:3.12

COPY deploy/requirements-lambda.txt .
RUN pip install --no-cache-dir -r requirements-lambda.txt

# Application code + the committed model (also loadable from S3, see infer.py).
COPY src/ ${LAMBDA_TASK_ROOT}/src/
COPY api/ ${LAMBDA_TASK_ROOT}/api/
COPY models/ppg_hr.onnx models/ppg_hr.json ${LAMBDA_TASK_ROOT}/models/
COPY deploy/lambda_handler.py ${LAMBDA_TASK_ROOT}/

# module.function that Lambda invokes.
CMD ["lambda_handler.handler"]
