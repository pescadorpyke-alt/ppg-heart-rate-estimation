<#
  Remove all AWS resources created by deploy.ps1, so nothing can incur cost.
  Run from project root:  powershell -ExecutionPolicy Bypass -File deploy\teardown.ps1
#>
[CmdletBinding()]
param(
  [string]$Region   = (aws configure get region),
  [string]$Repo     = "ppg-hr-api",
  [string]$Function = "ppg-hr-api",
  [string]$Bucket   = "",
  [string]$ModelKey = "models/ppg_hr.onnx"
)
$ErrorActionPreference = "Continue"
function Step($m) { Write-Host "`n=== $m ===" -ForegroundColor Yellow }

$AccountId = (aws sts get-caller-identity --query Account --output text)
if (-not $Bucket) { $Bucket = "ppg-hr-model-$AccountId-$Region" }
$Role = "$Function-role"

Step "Delete Function URL + Lambda"
aws lambda delete-function-url-config --function-name $Function 2>$null
aws lambda delete-function --function-name $Function 2>$null

Step "Delete IAM role"
aws iam delete-role-policy --role-name $Role --policy-name s3-model-read 2>$null
aws iam detach-role-policy --role-name $Role --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole 2>$null
aws iam delete-role --role-name $Role 2>$null

Step "Delete S3 object + bucket"
aws s3 rm "s3://$Bucket/$ModelKey" 2>$null
aws s3api delete-bucket --bucket $Bucket 2>$null

Step "Delete ECR repo (and images)"
aws ecr delete-repository --repository-name $Repo --force 2>$null

Step "DONE — all resources removed"
