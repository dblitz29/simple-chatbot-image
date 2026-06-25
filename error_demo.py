"""Deliberately trigger and gracefully handle a Bedrock error, captured as an
error span in Datadog LLM Observability.

Run:  python error_demo.py
Then in Datadog -> LLM Observability look for the `error_handling_demo` trace;
its child LLM span is marked errored, and ml_obs.trace.error increments.
"""
import io
import os

import boto3
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

from app.observability import setup_llm_observability
from ddtrace.llmobs import LLMObs
from ddtrace.llmobs.decorators import workflow

setup_llm_observability()

REGION = os.getenv("AWS_REGION", "ap-southeast-3").strip()
client = boto3.client("bedrock-runtime", region_name=REGION)

buf = io.BytesIO()
Image.new("RGB", (8, 8), (200, 50, 50)).save(buf, format="PNG")
img = buf.getvalue()

# An invalid model id -> deterministic ValidationException (no flapping).
BAD_MODEL = "anthropic.claude-this-model-does-not-exist-v1:0"


@workflow(name="error_handling_demo")
def run():
    try:
        client.converse(
            modelId=BAD_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"image": {"format": "png", "source": {"bytes": img}}},
                        {"text": "describe"},
                    ],
                }
            ],
            inferenceConfig={"maxTokens": 10},
        )
        return "unexpected success"
    except Exception as exc:
        # App handles the failure gracefully and records it on the trace.
        LLMObs.annotate(
            input_data=f"invoke {BAD_MODEL}",
            output_data={"handled_error": type(exc).__name__, "detail": str(exc)[:200]},
        )
        print("Error handled gracefully:", type(exc).__name__)
        print(str(exc)[:160])
        return "handled"


run()
LLMObs.flush()
print("\nError trace flushed. Datadog -> LLM Observability -> 'error_handling_demo'.")
