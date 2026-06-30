<#
  One-shot deploy of the PPG-HR API to AWS Lambda (container image) + Function URL.

  Prereqs (see deploy/README.md):
    - AWS CLI v2 configured (`aws configure`) with a region
    - Docker Desktop running
    - Trained checkpoint at D:\ppg-dalia-hr\checkpoints\best.pt

  Run from the PROJECT ROOT:
    powershell -ExecutionPolicy Bypass -File deploy\deploy.ps1

  Idempotent: safe to re-run; it updates existing resources.
#>
[CmdletBinding()]
param(
  [string]$Region      = (aws configure get region),
  [string]$Repo        = "ppg-hr-api",
  [string]$Function    = "ppg-hr-api",
  [string]$Bucket      = "",          # auto-generated if empty
  [string]$ModelKey    = "models/best.pt",
  [string]$ModelPath   = "D:\ppg-dalia-hr\checkpoints\best.pt",
  [int]   $MemoryMB    = 2048,
  [int]   $TimeoutSec  = 30
)
$ErrorActionPreference = "Stop"
function Step($m) { Write-Host "`n=== $m ===" -ForegroundColor Cyan }

if (-not $Region) { throw "No AWS region. Run 'aws configure set region us-east-1' (or pass -Region)." }
if (-not (Test-Path $ModelPath)) { throw "Checkpoint not found: $ModelPath. Train first (python -m src.train)." }

$AccountId = (aws sts get-caller-identity --query Account --output text)
if (-not $Bucket) { $Bucket = "ppg-hr-model-$AccountId-$Region" }
$Role    = "$Function-role"
$EcrUri  = "$AccountId.dkr.ecr.$Region.amazonaws.com/$Repo"
$ImageTag = "$EcrUri`:latest"
Write-Host "Account=$AccountId Region=$Region Bucket=$Bucket"

# --- 1. Docker image -----------------------------------------------------------
Step "Build image (linux/amd64)"
docker build --platform linux/amd64 -f deploy\Dockerfile -t $Repo .

# --- 2. ECR push ---------------------------------------------------------------
Step "Ensure ECR repo + push"
aws ecr describe-repositories --repository-names $Repo 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) { aws ecr create-repository --repository-name $Repo | Out-Null }
aws ecr get-login-password --region $Region | docker login --username AWS --password-stdin "$AccountId.dkr.ecr.$Region.amazonaws.com"
docker tag "$Repo`:latest" $ImageTag
docker push $ImageTag

# --- 3. S3 bucket + model ------------------------------------------------------
Step "Ensure S3 bucket + upload model"
aws s3api head-bucket --bucket $Bucket 2>$null
if ($LASTEXITCODE -ne 0) {
  if ($Region -eq "us-east-1") { aws s3api create-bucket --bucket $Bucket | Out-Null }
  else { aws s3api create-bucket --bucket $Bucket --create-bucket-configuration LocationConstraint=$Region | Out-Null }
}
aws s3 cp $ModelPath "s3://$Bucket/$ModelKey"

# --- 4. IAM role ---------------------------------------------------------------
Step "Ensure IAM role"
$RoleArn = (aws iam get-role --role-name $Role --query Role.Arn --output text 2>$null)
if ($LASTEXITCODE -ne 0) {
  $RoleArn = (aws iam create-role --role-name $Role `
      --assume-role-policy-document file://deploy/iam-trust-policy.json `
      --query Role.Arn --output text)
  aws iam attach-role-policy --role-name $Role `
      --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole | Out-Null
  Write-Host "Waiting 10s for role propagation..."; Start-Sleep -Seconds 10
}
# Inline S3 read policy (scoped to this bucket).
(Get-Content deploy/iam-s3-policy.json) -replace "__BUCKET__", $Bucket | Set-Content "$env:TEMP\ppg-s3-policy.json"
aws iam put-role-policy --role-name $Role --policy-name s3-model-read `
    --policy-document "file://$env:TEMP/ppg-s3-policy.json"

# --- 5. Lambda function --------------------------------------------------------
Step "Create/update Lambda function"
$envVars = "Variables={MODEL_S3_BUCKET=$Bucket,MODEL_S3_KEY=$ModelKey,MODEL_LOCAL_PATH=/tmp/best.pt}"
aws lambda get-function --function-name $Function 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
  aws lambda create-function --function-name $Function `
    --package-type Image --code ImageUri=$ImageTag `
    --role $RoleArn --memory-size $MemoryMB --timeout $TimeoutSec `
    --environment $envVars | Out-Null
} else {
  aws lambda update-function-code --function-name $Function --image-uri $ImageTag | Out-Null
  aws lambda wait function-updated --function-name $Function
  aws lambda update-function-configuration --function-name $Function `
    --memory-size $MemoryMB --timeout $TimeoutSec --environment $envVars | Out-Null
}
aws lambda wait function-updated --function-name $Function

# --- 6. Public Function URL ----------------------------------------------------
Step "Ensure public Function URL"
$Url = (aws lambda get-function-url-config --function-name $Function --query FunctionUrl --output text 2>$null)
if ($LASTEXITCODE -ne 0) {
  $Url = (aws lambda create-function-url-config --function-name $Function `
      --auth-type NONE --query FunctionUrl --output text)
  aws lambda add-permission --function-name $Function --statement-id public-url `
      --action lambda:InvokeFunctionUrl --principal "*" --function-url-auth-type NONE | Out-Null
}

Step "DONE"
Write-Host "Public endpoint: $Url" -ForegroundColor Green
Write-Host "Test:"
Write-Host "  curl ${Url}health"
Write-Host "  curl -X POST ${Url}predict -H `"Content-Type: application/json`" -d `@api/example_payload.json"
