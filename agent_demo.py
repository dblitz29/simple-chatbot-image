"""Run the instrumented agent once and ship a full trace to Datadog.

Usage:
    python agent_demo.py            # uses a generated sample image
    python agent_demo.py photo.jpg  # uses your own image

Then open Datadog -> LLM Observability -> ML app, and look for the
`image_analysis_agent` workflow trace (workflow > task > tool > llm spans).
"""
import io
import sys

from dotenv import load_dotenv

load_dotenv()

from app.observability import setup_llm_observability
from app.agent import run_agent

setup_llm_observability()


def _load_image():
    if len(sys.argv) > 1:
        path = sys.argv[1]
        with open(path, "rb") as fh:
            data = fh.read()
        ext = path.rsplit(".", 1)[-1].lower()
        media = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
        return data, media
    from PIL import Image

    buf = io.BytesIO()
    img = Image.new("RGB", (320, 200), (40, 90, 200))
    # a little contrast block so there is something to describe
    for x in range(120, 200):
        for y in range(60, 140):
            img.putpixel((x, y), (240, 210, 40))
    img.save(buf, format="PNG")
    return buf.getvalue(), "image/png"


image_bytes, media_type = _load_image()
result = run_agent(image_bytes, media_type, "Describe this image and save a report.")

print("\n=== AGENT RESULT ===")
print("Model:", result["model"])
print("Tool calls:", [t["tool"] for t in result["tool_calls"]])
print("Answer:", result["answer"])

from ddtrace.llmobs import LLMObs

LLMObs.flush()
print("\nTrace flushed. Check Datadog LLM Observability -> workflow 'image_analysis_agent'.")
