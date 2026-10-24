# Deploy via AWS CloudShell (no local Docker / no local credentials)

CloudShell is a browser terminal that's already logged into your AWS account and
ships Docker + AWS CLI. This is the easiest path to a real endpoint.

## Steps

1. **Sign in** to the AWS Console and pick your region (top-right). `us-east-1`
   is a safe default. *The endpoint will be created in whatever region you select.*

2. **Open CloudShell** — the terminal icon `>_` in the top navigation bar. Wait
   for it to start.

3. **Upload the bundle:** in CloudShell, **Actions → Upload file**, and choose
   `ppg-hr-cloudshell-bundle.zip` (in the project root on your machine).

4. **Run the deploy** (paste into CloudShell):

   ```bash
   mkdir -p ppg && cd ppg
   unzip -o ../ppg-hr-cloudshell-bundle.zip
   bash deploy/deploy.sh
   ```

   It builds the image, pushes to ECR, uploads the model to S3, creates the Lambda
   + a public Function URL, and prints the URL. First run takes a few minutes
   (Docker build + image push).

5. **Test** (the script prints these with your real URL):

   ```bash
   curl ${URL}health
   curl -X POST ${URL}predict -H 'Content-Type: application/json' -d @api/example_payload.json
   ```

   The first call is a cold start (a few seconds); subsequent calls are fast.

## If the build fails with "no space left on device"

CloudShell has limited disk. Free it and retry:

```bash
docker system prune -af
bash deploy/deploy.sh
```

If it still won't fit, fall back to local Docker Desktop with `deploy/deploy.ps1`.

## Tear down (from CloudShell, to avoid any cost)

The repo's `deploy/teardown.ps1` is PowerShell; in CloudShell delete via CLI:

```bash
aws lambda delete-function-url-config --function-name ppg-hr-api
aws lambda delete-function --function-name ppg-hr-api
aws iam delete-role-policy --role-name ppg-hr-api-role --policy-name s3-model-read
aws iam detach-role-policy --role-name ppg-hr-api-role --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
aws iam delete-role --role-name ppg-hr-api-role
ACCOUNT=$(aws sts get-caller-identity --query Account --output text); REGION=${AWS_REGION}
aws s3 rm s3://ppg-hr-model-${ACCOUNT}-${REGION}/models/ppg_hr.onnx
aws s3api delete-bucket --bucket ppg-hr-model-${ACCOUNT}-${REGION}
aws ecr delete-repository --repository-name ppg-hr-api --force
```
