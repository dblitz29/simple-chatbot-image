"""Simple image analysis web app powered by AWS Bedrock."""
import os

from dotenv import load_dotenv

# Load .env before anything reads environment variables.
load_dotenv()

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request

from app.bedrock import analyze_image
from app.agent import run_agent
from app.observability import setup_llm_observability

# Enable Datadog LLM Observability before any Bedrock client is created so the
# boto3/bedrock calls get auto-instrumented.
setup_llm_observability()

app = FastAPI(title="Image Analysis with Bedrock")
templates = Jinja2Templates(directory="app/templates")

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
MAX_BYTES = 5 * 1024 * 1024  # 5 MB


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    prompt: str = Form("Describe this image in detail."),
):
    if image.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {image.content_type}. "
            f"Allowed: {', '.join(sorted(ALLOWED_TYPES))}",
        )

    data = await image.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=400, detail="Image too large (max 5 MB).")

    try:
        result = analyze_image(data, image.content_type, prompt)
    except Exception as exc:  # surface Bedrock/credential errors cleanly
        raise HTTPException(status_code=502, detail=f"Bedrock error: {exc}")

    return JSONResponse({"analysis": result})


@app.post("/agent")
async def agent(
    image: UploadFile = File(...),
    prompt: str = Form("Describe this image and save a report."),
):
    """Run the instrumented agentic workflow (workflow > task > tool > llm)."""
    if image.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {image.content_type}.",
        )

    data = await image.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=400, detail="Image too large (max 5 MB).")

    try:
        result = run_agent(data, image.content_type, prompt)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Agent error: {exc}")

    return JSONResponse(result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
