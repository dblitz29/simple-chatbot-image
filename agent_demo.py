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
result = run_agent(
    image_bytes,
    media_type,
    "Assess this vehicle damage photo for an insurance reimbursement claim.",
)

print("\n=== CLAIM ASSESSMENT ===")
print("Model:", result["model"])
print("Tool calls:", [t["tool"] for t in result["tool_calls"]])
if result.get("decision"):
    d = result["decision"]
    print(f"Decision: {d['decision']}  | payout ${d['approved_payout_usd']}")
    print(f"Damage: {d['damage_type']} ({d['severity']}), est ${d['estimated_cost_usd']}")
print("Summary:", result["answer"])

from ddtrace.llmobs import LLMObs

LLMObs.flush()
print("\nTrace flushed. Check Datadog LLM Observability -> workflow 'insurance_claim_agent'.")
