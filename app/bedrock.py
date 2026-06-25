"""Bedrock client helper for image analysis using Claude vision models."""
import base64
import json
import os

import boto3

# Region & model are configurable via environment variables.
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
# Default to Claude 3.5 Sonnet (supports vision). Override with BEDROCK_MODEL_ID.
MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0")

_client = None


def get_client():
    """Lazily create a shared Bedrock runtime client."""
    global _client
    if _client is None:
        _client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    return _client


def analyze_image(image_bytes: bytes, media_type: str, prompt: str) -> str:
    """Send an image + prompt to Bedrock and return the text analysis."""
    encoded = base64.standard_b64encode(image_bytes).decode("utf-8")

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": encoded,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }

    response = get_client().invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json",
    )

    payload = json.loads(response["body"].read())
    return payload["content"][0]["text"]
