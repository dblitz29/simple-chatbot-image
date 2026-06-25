# Image Analysis with AWS Bedrock

A simple FastAPI web app that analyzes uploaded images using AWS Bedrock (Claude vision models). Upload an image, give a prompt, get an analysis. Ready to dockerize and deploy to AWS.

## Features

- Clean single-page web UI with image preview
- `/analyze` JSON API endpoint
- `/health` health check (for load balancers)
- Configurable region and model via env vars
- Docker + docker-compose included

## Prerequisites

1. An AWS account with **Bedrock model access enabled** for the Claude model you want to use (Bedrock console → Model access).
2. AWS credentials with permission to call `bedrock:InvokeModel`.

## Run locally

```bash
pip install -r requirements.txt

# set credentials (or use an AWS profile / IAM role)
set AWS_REGION=us-east-1
set AWS_ACCESS_KEY_ID=...
set AWS_SECRET_ACCESS_KEY=...

uvicorn app.main:app --reload
```

Open http://localhost:8000

## Run with Docker

```bash
docker build -t bedrock-image-analysis .

docker run -p 8000:8000 ^
  -e AWS_REGION=us-east-1 ^
  -e AWS_ACCESS_KEY_ID=... ^
  -e AWS_SECRET_ACCESS_KEY=... ^
  bedrock-image-analysis
```

Or with docker-compose (reads vars from a `.env` file):

```bash
docker compose up --build
```

## API

```bash
curl -X POST http://localhost:8000/analyze \
  -F "image=@photo.jpg" \
  -F "prompt=What objects are in this image?"
```

Response:

```json
{ "analysis": "The image shows ..." }
```

## Deploy to AWS (simple & fast)

The fastest path is **App Runner** (no servers to manage) or **ECS Fargate**.

### Option A — AWS App Runner (recommended for speed)

1. Push the image to ECR:

   ```bash
   aws ecr create-repository --repository-name bedrock-image-analysis
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <acct>.dkr.ecr.us-east-1.amazonaws.com
   docker build -t bedrock-image-analysis .
   docker tag bedrock-image-analysis:latest <acct>.dkr.ecr.us-east-1.amazonaws.com/bedrock-image-analysis:latest
   docker push <acct>.dkr.ecr.us-east-1.amazonaws.com/bedrock-image-analysis:latest
   ```

2. In the App Runner console, create a service from that ECR image.
   - Port: `8000`
   - Health check path: `/health`
   - Attach an **instance role** with `bedrock:InvokeModel` permission (no static keys needed).
   - Set env var `AWS_REGION` to your Bedrock region.

### Option B — ECS Fargate

Use the same ECR image. Define a task with port `8000`, attach a **task role** granting `bedrock:InvokeModel`, and front it with an ALB pointing health checks at `/health`.

### Minimal IAM policy for the runtime role

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "bedrock:InvokeModel",
      "Resource": "*"
    }
  ]
}
```

> In production, use an IAM role (App Runner instance role / ECS task role) instead of static access keys.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `AWS_REGION` | `us-east-1` | Bedrock region |
| `BEDROCK_MODEL_ID` | `anthropic.claude-3-5-sonnet-20240620-v1:0` | Vision-capable model |
| `PORT` | `8000` | Local run port |
