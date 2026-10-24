# Deploy — AWS Lambda (container image) + Function URL

The inference API runs on **AWS Lambda** as a container image, exposed via a
**Lambda Function URL** (a public HTTPS endpoint). The trained model lives in
**S3** and is downloaded to `/tmp` on cold start.

## Why Lambda (not EC2)?

| | EC2 t2.micro | **Lambda + Function URL** |
|---|---|---|
| Free tier | 750 h/mo for **12 months**, then ~$8.5/mo always-on | 1M req + 400k GB-s/mo **forever** |
| Idle cost | Billed while running | **$0 (scales to zero)** |
| Maintenance | OS patching, securing ports, can crash | **None** |
| Latency | Always warm | Cold start ~3–8 s, then warm |

For a portfolio demo endpoint that won't be actively monitored, Lambda wins:
nothing to patch, no server to leave running, no surprise bill. The only cost is
a few seconds of cold-start latency on the first request after idle — irrelevant
here.

> **Function URL vs. API Gateway:** a Function URL is the simplest public HTTPS
> entry point and is included free. API Gateway adds a configurable layer
> (throttling, API keys, custom domains) that this demo doesn't need. The
> Dockerized app works behind either — to use API Gateway instead, create an
> HTTP API with a Lambda proxy integration to the same function.

## Prerequisites

1. **AWS CLI v2** configured: `aws configure` (set an access key + region, e.g. `us-east-1`).
2. **Docker Desktop** running.
3. The committed model `models/ppg_hr.onnx` (run the script from the project root).

## Deploy

From the **project root**:

```powershell
powershell -ExecutionPolicy Bypass -File deploy\deploy.ps1
```

The script (idempotent — safe to re-run) will:

1. Build the `linux/amd64` image from [`deploy/Dockerfile`](Dockerfile).
2. Create an **ECR** repo and push the image.
3. Create an **S3** bucket (`ppg-hr-model-<account>-<region>`) and upload the model.
4. Create an **IAM role** (basic execution + `s3:GetObject` scoped to that bucket).
5. Create/update the **Lambda** function (container image, 2048 MB, 30 s timeout)
   with env vars `MODEL_S3_BUCKET`, `MODEL_S3_KEY`, `MODEL_LOCAL_PATH=/tmp/ppg_hr.onnx`.
6. Create a public **Function URL** and print it.

Then test (the script prints these with your real URL):

```bash
curl https://<id>.lambda-url.<region>.on.aws/health
curl -X POST https://<id>.lambda-url.<region>.on.aws/predict \
  -H "Content-Type: application/json" -d @api/example_payload.json
```

## Tear down (avoid any cost)

```powershell
powershell -ExecutionPolicy Bypass -File deploy\teardown.ps1
```

Removes the Function URL, Lambda, IAM role, S3 object+bucket, and ECR repo.

## Notes / gotchas

- **Only `/tmp` is writable on Lambda** — hence `MODEL_LOCAL_PATH=/tmp/ppg_hr.onnx`.
- **Image size:** no PyTorch in the image; the API runs `models/ppg_hr.onnx` with
  ONNX Runtime (DECISIONS.md D23). The image has not been rebuilt since that change.
- **Cold start:** importing ONNX Runtime + loading the model takes a moment on the
  first call; subsequent calls (warm) respond in well under a second.
- **Python versions:** the image uses Python 3.12 (Lambda runtime); the model is an
  ONNX file, independent of the Python or PyTorch version used to train it.
- **Cost ceiling:** even with generous demo traffic this stays within the
  always-free Lambda tier; S3 storage for a ~186 KB model is negligible.
