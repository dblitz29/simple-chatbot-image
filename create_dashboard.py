"""Create the LLM Observability dashboard in Datadog via the API (no Terraform).

Reads DD_API_KEY / DD_APP_KEY / DD_SITE from .env and posts a dashboard whose
widgets match shared/dashboard.tf. Prints the live dashboard URL.
"""
import json
import os
import urllib.request

from dotenv import load_dotenv

load_dotenv()

SITE = os.getenv("DD_SITE", "datadoghq.com").strip()
API_KEY = os.getenv("DD_API_KEY", "").strip()
APP_KEY = os.getenv("DD_APP_KEY", "").strip()
ML_APP = os.getenv("DD_LLMOBS_ML_APP", "image-analysis").strip()

dashboard = {
    "title": "Jakarta Hackathon — LLM Observability",
    "layout_type": "ordered",
    "description": "Image analysis app LLM metrics (rebuilt via API).",
    "widgets": [
        {
            "definition": {
                "type": "timeseries",
                "title": "LLM Request Rate by Model",
                "requests": [
                    {
                        "q": "sum:ml_obs.trace{*} by {model_name}.as_rate()",
                        "display_type": "line",
                    }
                ],
            }
        },
        {
            "definition": {
                "type": "timeseries",
                "title": "P95 Duration",
                "requests": [
                    {"q": "p95:ml_obs.trace.duration{*}", "display_type": "line"}
                ],
            }
        },
        {
            "definition": {
                "type": "query_value",
                "title": "Total Tokens / Cost (last 1h)",
                "autoscale": True,
                "requests": [
                    {
                        "q": "sum:ml_obs.span.llm.total.cost{*}.as_count()",
                        "aggregator": "sum",
                    }
                ],
            }
        },
        {
            "definition": {
                "type": "timeseries",
                "title": "LLM Error Rate",
                "requests": [
                    {
                        "q": "sum:ml_obs.trace.error{*}.as_count()",
                        "display_type": "bars",
                    }
                ],
            }
        },
    ],
}

url = f"https://api.{SITE}/api/v1/dashboard"
req = urllib.request.Request(
    url,
    data=json.dumps(dashboard).encode("utf-8"),
    method="POST",
    headers={
        "Content-Type": "application/json",
        "DD-API-KEY": API_KEY,
        "DD-APPLICATION-KEY": APP_KEY,
    },
)

try:
    with urllib.request.urlopen(req) as resp:
        body = json.loads(resp.read())
    print("Created dashboard id:", body.get("id"))
    print("URL: https://app." + SITE + body.get("url", ""))
except urllib.error.HTTPError as e:
    print("HTTP", e.code, e.read().decode())
