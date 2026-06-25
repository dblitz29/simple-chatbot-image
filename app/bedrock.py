"""Bedrock client helper for image analysis using the Converse API.

The hackathon account intermittently returns a "channel program accounts"
ValidationException for vision calls (it flaps between models). To stay
reliable we retry each model a few times and fall back across a model list,
keeping the configured model as the preferred one.
"""
import os
import time

import boto3
from botocore.exceptions import ClientError

AWS_REGION = os.getenv("AWS_REGION", "us-east-1").strip()
MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "amazon.nova-pro-v1:0").strip()

# Preferred model first, then fallbacks tried in order when it errors.
_DEFAULT_FALLBACKS = (
    "apac.amazon.nova-pro-v1:0,amazon.nova-lite-v1:0,"
    "anthropic.claude-haiku-4-5-20251001-v1:0"
)
FALLBACK_MODELS = [
    m.strip()
    for m in os.getenv("BEDROCK_FALLBACK_MODELS", _DEFAULT_FALLBACKS).split(",")
    if m.strip()
]

MAX_RETRIES_PER_MODEL = int(os.getenv("BEDROCK_MAX_RETRIES", "2"))

_FORMAT_BY_MEDIA_TYPE = {
    "image/jpeg": "jpeg",
    "image/jpg": "jpeg",
    "image/png": "png",
    "image/gif": "gif",
    "image/webp": "webp",
}

# Errors worth retrying / failing over on (transient or flapping access).
_RETRYABLE = {
    "ValidationException",
    "ThrottlingException",
    "ModelNotReadyException",
    "ServiceUnavailableException",
    "InternalServerException",
}

_client = None


def get_client():
    global _client
    if _client is None:
        _client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    return _client


def _model_order():
    """Configured model first, then fallbacks (de-duplicated)."""
    seen, order = set(), []
    for m in [MODEL_ID, *FALLBACK_MODELS]:
        if m and m not in seen:
            seen.add(m)
            order.append(m)
    return order


def analyze_image(image_bytes: bytes, media_type: str, prompt: str) -> str:
    image_format = _FORMAT_BY_MEDIA_TYPE.get(media_type)
    if image_format is None:
        raise ValueError(f"Unsupported media type: {media_type}")

    content = [
        {"image": {"format": image_format, "source": {"bytes": image_bytes}}},
        {"text": prompt},
    ]

    last_error = None
    for model_id in _model_order():
        for attempt in range(MAX_RETRIES_PER_MODEL + 1):
            try:
                response = get_client().converse(
                    modelId=model_id,
                    messages=[{"role": "user", "content": content}],
                    inferenceConfig={"maxTokens": 1024, "temperature": 0.2},
                )
                return response["output"]["message"]["content"][0]["text"]
            except ClientError as exc:
                last_error = exc
                code = exc.response.get("Error", {}).get("Code", "")
                if code not in _RETRYABLE:
                    raise
                # brief backoff before retrying the same model
                if attempt < MAX_RETRIES_PER_MODEL:
                    time.sleep(0.5 * (attempt + 1))
        # exhausted retries for this model -> try the next fallback

    raise last_error
