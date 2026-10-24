#!/usr/bin/env bash
# Deploy the PPG-HR API to AWS Lambda (container) + Function URL — for AWS CloudShell.
#
# CloudShell is pre-authenticated and ships docker + aws cli, so no local Docker
# or local credentials are needed. Run from the unzipped bundle root:
#   bash deploy/deploy.sh
#
# Idempotent: safe to re-run (updates existing resources).
set -euo pipefail

REPO="ppg-hr-api"
FUNCTION="ppg-hr-api"
MODEL_KEY="models/ppg_hr.onnx"
MODEL_PATH="./models/ppg_hr.onnx"   # committed with the code
MEMORY_MB=2048
TIMEOUT_SEC=30

step(){ echo -e "\n=== $* ==="; }

REGION="${AWS_REGION:-$(aws configure get region || true)}"
[ -n "$REGION" ] || { echo "No region. Run: export AWS_REGION=us-east-1"; exit 1; }
[ -f "$MODEL_PATH" ] || { echo "Model not found at $MODEL_PATH (run from the project root)."; exit 1; }

ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
BUCKET="ppg-hr-model-${ACCOUNT}-${REGION}"
ROLE="${FUNCTION}-role"
ECR="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REPO}"
IMAGE="${ECR}:latest"
echo "Account=$ACCOUNT Region=$REGION Bucket=$BUCKET"

step "Build image (linux/amd64)"
docker build --platform linux/amd64 -f deploy/Dockerfile -t "$REPO" .

step "Ensure ECR repo + push"
aws ecr describe-repositories --repository-names "$REPO" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "$REPO" >/dev/null
aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"
docker tag "${REPO}:latest" "$IMAGE"
docker push "$IMAGE"

step "Ensure S3 bucket + upload model"
if ! aws s3api head-bucket --bucket "$BUCKET" >/dev/null 2>&1; then
  if [ "$REGION" = "us-east-1" ]; then
    aws s3api create-bucket --bucket "$BUCKET" >/dev/null
  else
    aws s3api create-bucket --bucket "$BUCKET" \
      --create-bucket-configuration LocationConstraint="$REGION" >/dev/null
  fi
fi
aws s3 cp "$MODEL_PATH" "s3://${BUCKET}/${MODEL_KEY}"

step "Ensure IAM role"
if ! ROLE_ARN="$(aws iam get-role --role-name "$ROLE" --query Role.Arn --output text 2>/dev/null)"; then
  ROLE_ARN="$(aws iam create-role --role-name "$ROLE" \
    --assume-role-policy-document file://deploy/iam-trust-policy.json \
    --query Role.Arn --output text)"
  aws iam attach-role-policy --role-name "$ROLE" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
  echo "Waiting 10s for role propagation..."; sleep 10
fi
sed "s/__BUCKET__/${BUCKET}/" deploy/iam-s3-policy.json > /tmp/ppg-s3-policy.json
aws iam put-role-policy --role-name "$ROLE" --policy-name s3-model-read \
  --policy-document file:///tmp/ppg-s3-policy.json

step "Create/update Lambda function"
ENVV="Variables={MODEL_S3_BUCKET=${BUCKET},MODEL_S3_KEY=${MODEL_KEY},MODEL_LOCAL_PATH=/tmp/ppg_hr.onnx}"
if ! aws lambda get-function --function-name "$FUNCTION" >/dev/null 2>&1; then
  aws lambda create-function --function-name "$FUNCTION" \
    --package-type Image --code ImageUri="$IMAGE" \
    --role "$ROLE_ARN" --memory-size "$MEMORY_MB" --timeout "$TIMEOUT_SEC" \
    --environment "$ENVV" >/dev/null
else
  aws lambda update-function-code --function-name "$FUNCTION" --image-uri "$IMAGE" >/dev/null
  aws lambda wait function-updated --function-name "$FUNCTION"
  aws lambda update-function-configuration --function-name "$FUNCTION" \
    --memory-size "$MEMORY_MB" --timeout "$TIMEOUT_SEC" --environment "$ENVV" >/dev/null
fi
aws lambda wait function-updated --function-name "$FUNCTION"

step "Ensure public Function URL"
if ! URL="$(aws lambda get-function-url-config --function-name "$FUNCTION" --query FunctionUrl --output text 2>/dev/null)"; then
  URL="$(aws lambda create-function-url-config --function-name "$FUNCTION" \
    --auth-type NONE --query FunctionUrl --output text)"
  aws lambda add-permission --function-name "$FUNCTION" --statement-id public-url \
    --action lambda:InvokeFunctionUrl --principal "*" --function-url-auth-type NONE >/dev/null
fi

step "DONE"
echo "Public endpoint: $URL"
echo "Test:"
echo "  curl ${URL}health"
echo "  curl -X POST ${URL}predict -H 'Content-Type: application/json' -d @api/example_payload.json"
