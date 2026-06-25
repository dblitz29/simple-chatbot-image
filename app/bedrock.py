"""Bedrock client helper for image analysis using the Converse API.

The Converse API uses one unified schema across all modern Bedrock models
(Claude 3.x/4.x, etc.) and natively supports images, so we avoid model-specific
request bodies.
"""
import os

import boto3

# Region & model are configurable via environment variables.
AWS_REGION = os.getenv("AWS_REGION", "us-east-1").strip()
MODEL_ID = os.getenv(
    "BEDROCK_MODEL_ID", "anthropic.claude-haiku-4-5-20251001-v1:0"
).strip()

# Map HTTP content types to the format names Converse expects.
_FORMAT_BY_MEDIA_TYPE = {
    "image/jpeg": "jpeg",
    "image/jpg": "jpeg",
    "image/png": "png",
    "image/gif": "gif",
    "image/webp": "webp",
}

_client = None


def get_client():
    """Lazily create a shared Bedrock runtime client."""
    global _client
    if _client is None:
        _client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    return _client


def analyze_image(image_bytes: bytes, media_type: str, prompt: str) -> str:
    """Send an image + prompt to Bedrock via Converse and return the text."""
    image_format = _FORMAT_BY_MEDIA_TYPE.get(media_type)
    if image_format is None:
        raise ValueError(f"Unsupported media type: {media_type}")

    response = get_client().converse(
        modelId=MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": [
                    {"image": {"format": image_format, "source": {"bytes": image_bytes}}},
                    {"text": prompt},
                ],
            }
        ],
        inferenceConfig={"maxTokens": 1024, "temperature": 0.2},
    )

    return response["output"]["message"]["content"][0]["text"]
