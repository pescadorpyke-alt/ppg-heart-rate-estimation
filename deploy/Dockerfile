# AWS Lambda container image for the PPG-HR inference API.
# Build context = project root:  docker build -f deploy/Dockerfile -t ppg-hr-api .
#
# Python 3.12 base (AWS Lambda runtime). The model checkpoint is portable across
# Python versions, so training on 3.14 locally and serving on 3.12 here is fine.
FROM public.ecr.aws/lambda/python:3.12

# CPU-only PyTorch first, from the dedicated CPU index (keeps the image small;
# the default PyPI wheel would drag in CUDA and balloon the image by GBs).
RUN pip install --no-cache-dir torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu

# Remaining inference deps.
COPY deploy/requirements-lambda.txt .
RUN pip install --no-cache-dir -r requirements-lambda.txt

# Application code.
COPY src/ ${LAMBDA_TASK_ROOT}/src/
COPY api/ ${LAMBDA_TASK_ROOT}/api/
COPY deploy/lambda_handler.py ${LAMBDA_TASK_ROOT}/

# module.function that Lambda invokes.
CMD ["lambda_handler.handler"]
