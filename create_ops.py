"""Create Ops-ready Datadog resources: a Monitor (with runbook + on-call alert)
and an SLO, both based on the app's LLM Observability metrics.

Run once:  python create_ops.py
Requires DD_API_KEY / DD_APP_KEY with monitors_write + slos_write scopes.
Override the on-call handle with DD_ONCALL (e.g. "@slack-oncall" or "@pagerduty-llm").
"""
import json
import os
import urllib.error
import urllib.request

from dotenv import load_dotenv

load_dotenv()

SITE = os.getenv("DD_SITE", "datadoghq.com").strip()
API_KEY = os.getenv("DD_API_KEY", "").strip()
APP_KEY = os.getenv("DD_APP_KEY", "").strip()
ML_APP = os.getenv("DD_LLMOBS_ML_APP", "image-analysis").strip()
ONCALL = os.getenv("DD_ONCALL", "@here").strip()

HEADERS = {
    "Content-Type": "application/json",
    "DD-API-KEY": API_KEY,
    "DD-APPLICATION-KEY": APP_KEY,
}


def _post(path, payload):
    req = urllib.request.Request(
        f"https://api.{SITE}{path}",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers=HEADERS,
    )
    try:
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read()), None
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.read().decode()[:200]}"


# ── 1) Monitor: alert on LLM error rate, with a runbook + on-call notify ────
runbook = (
    "## Runbook: LLM Error Rate High\\n"
    "1. Check the LLM Observability trace explorer for failing spans "
    f"(ml_app:{ML_APP}).\\n"
    "2. Common cause: Bedrock 'channel program' / throttling — the app "
    "auto-retries and falls back across models; confirm fallbacks are healthy.\\n"
    "3. Verify AWS creds + region (ap-southeast-3) and Bedrock model access.\\n"
    "4. If sustained, page the on-call owner and roll back the last deploy."
)

monitor = {
    "name": f"[{ML_APP}] LLM Error Rate High",
    "type": "query alert",
    "query": "sum(last_5m):sum:ml_obs.trace.error{*}.as_count() > 5",
    "message": (
        f"{{{{#is_alert}}}}LLM errors exceeded threshold for {ML_APP}.\\n\\n"
        f"{runbook}\\n\\nPaging on-call: {ONCALL}{{{{/is_alert}}}}\\n"
        f"{{{{#is_recovery}}}}LLM error rate recovered. {ONCALL}{{{{/is_recovery}}}}"
    ),
    "tags": [f"ml_app:{ML_APP}", "service:image-analysis", "team:on-call"],
    "options": {
        "thresholds": {"critical": 5, "warning": 2},
        "notify_no_data": False,
        "renotify_interval": 0,
    },
}

mon, err = _post("/api/v1/monitor", monitor)
if err:
    print("Monitor FAILED ->", err)
    monitor_id = None
else:
    monitor_id = mon["id"]
    print("Monitor created -> id", monitor_id)

# ── 2) SLO: 99% of LLM requests succeed over 7d/30d ─────────────────────────
if monitor_id:
    slo = {
        "type": "monitor",
        "name": f"[{ML_APP}] LLM Request Success SLO",
        "description": "99% of image-analysis LLM requests succeed (error-free).",
        "monitor_ids": [monitor_id],
        "thresholds": [
            {"timeframe": "7d", "target": 99.0, "warning": 99.5},
            {"timeframe": "30d", "target": 99.0, "warning": 99.5},
        ],
        "tags": [f"ml_app:{ML_APP}", "service:image-analysis"],
    }
    slo_resp, err = _post("/api/v1/slo", slo)
    if err:
        print("SLO FAILED ->", err)
    else:
        data = slo_resp.get("data", [{}])
        print("SLO created -> id", data[0].get("id") if data else slo_resp)
else:
    print("Skipping SLO (monitor was not created).")
