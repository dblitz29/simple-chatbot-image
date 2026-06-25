"""Generate a few real LLM traces to populate Datadog LLM Observability.

Safe to delete after running. Sends several small Bedrock calls so the
ml_obs.* metrics have visible data for the dashboard / judging.
"""
import io

from dotenv import load_dotenv

load_dotenv()

from app.observability import setup_llm_observability
from app.bedrock import analyze_image
from PIL import Image

setup_llm_observability()

colors = [(220, 60, 60), (60, 200, 90), (240, 200, 40), (90, 90, 230), (200, 200, 200)]
for i, rgb in enumerate(colors, 1):
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), rgb).save(buf, format="PNG")
    out = analyze_image(buf.getvalue(), "image/png", "Name the dominant color in 3 words max.")
    print(f"[{i}/{len(colors)}] {rgb} -> {out.strip()}")

from ddtrace.llmobs import LLMObs

LLMObs.flush()
print("Done. Traces flushed to Datadog.")
