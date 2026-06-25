"""Tools the agent can call during an image-analysis session.

Each tool receives the current image bytes (injected by the agent runner) plus
any arguments the model provided. Tools return plain JSON-serializable dicts.
"""
import io
import os
import re
from datetime import datetime, timezone

from PIL import Image

REPORTS_DIR = os.getenv("REPORTS_DIR", "reports")


# --- Tool specifications advertised to the model (Bedrock Converse format) ---
TOOL_SPECS = [
    {
        "toolSpec": {
            "name": "get_image_properties",
            "description": (
                "Get exact technical properties of the uploaded image: width, "
                "height, format, color mode, file size, and dominant colors. "
                "Use this when you need precise measurements instead of guessing."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "top_colors": {
                            "type": "integer",
                            "description": "How many dominant colors to return (1-10).",
                        }
                    },
                    "required": [],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "assess_claim",
            "description": (
                "Apply the insurance policy rules to decide an auto claim. Call "
                "this after you have classified the damage from the image. Returns "
                "the decision (APPROVE / MANUAL_REVIEW / REJECT) and estimated "
                "payout."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "damage_type": {
                            "type": "string",
                            "description": "e.g. dented_bumper, broken_windshield, scratch, none.",
                        },
                        "severity": {
                            "type": "string",
                            "description": "One of: none, minor, moderate, severe.",
                        },
                        "estimated_cost_usd": {
                            "type": "number",
                            "description": "Estimated repair cost in USD.",
                        },
                    },
                    "required": ["damage_type", "severity", "estimated_cost_usd"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "save_report",
            "description": (
                "Save the final analysis to a text report file so the user can "
                "keep it. Call this once the analysis is complete."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Short report title."},
                        "content": {"type": "string", "description": "Full report body."},
                    },
                    "required": ["title", "content"],
                }
            },
        }
    },
]


def _dominant_colors(img: Image.Image, count: int):
    """Return the most common colors as hex strings using a downscaled image."""
    small = img.convert("RGB").resize((64, 64))
    colors = small.getcolors(64 * 64) or []
    colors.sort(key=lambda c: c[0], reverse=True)
    return [
        "#{:02x}{:02x}{:02x}".format(r, g, b)
        for _, (r, g, b) in colors[:count]
    ]


def get_image_properties(image_bytes: bytes, top_colors: int = 5, **_):
    top_colors = max(1, min(int(top_colors or 5), 10))
    img = Image.open(io.BytesIO(image_bytes))
    return {
        "width": img.width,
        "height": img.height,
        "format": img.format,
        "mode": img.mode,
        "size_kb": round(len(image_bytes) / 1024, 1),
        "dominant_colors": _dominant_colors(img, top_colors),
    }


def assess_claim(
    image_bytes: bytes,
    damage_type: str = "unknown",
    severity: str = "minor",
    estimated_cost_usd: float = 0.0,
    **_,
):
    """Deterministic insurance policy decision based on the classified damage."""
    severity = (severity or "minor").lower().strip()
    try:
        cost = float(estimated_cost_usd or 0.0)
    except (TypeError, ValueError):
        cost = 0.0

    DEDUCTIBLE = 250.0
    REVIEW_THRESHOLD = 2000.0

    if severity in ("none", "no damage", "") or cost <= 0:
        decision = "REJECT"
        payout = 0.0
        rationale = "No assessable damage detected in the image."
    elif severity == "severe" or cost > REVIEW_THRESHOLD:
        decision = "MANUAL_REVIEW"
        payout = 0.0
        rationale = (
            f"High severity or cost above ${REVIEW_THRESHOLD:.0f} threshold; "
            "route to a human adjuster."
        )
    else:
        decision = "APPROVE"
        payout = round(max(cost - DEDUCTIBLE, 0.0), 2)
        rationale = (
            f"{severity.capitalize()} {damage_type} within auto-approval limits; "
            f"payout = cost ${cost:.0f} minus ${DEDUCTIBLE:.0f} deductible."
        )

    return {
        "damage_type": damage_type,
        "severity": severity,
        "estimated_cost_usd": cost,
        "decision": decision,
        "approved_payout_usd": payout,
        "deductible_usd": DEDUCTIBLE,
        "rationale": rationale,
    }


def save_report(image_bytes: bytes, title: str = "report", content: str = "", **_):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "report"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = os.path.join(REPORTS_DIR, f"{slug}-{stamp}.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"# {title}\n\n{content}\n")
    return {"saved_to": path, "bytes_written": os.path.getsize(path)}


# Map tool name -> callable
REGISTRY = {
    "get_image_properties": get_image_properties,
    "assess_claim": assess_claim,
    "save_report": save_report,
}


def run_tool(name: str, image_bytes: bytes, arguments: dict):
    """Execute a tool by name; never raises, returns an error dict instead."""
    fn = REGISTRY.get(name)
    if fn is None:
        return {"error": f"Unknown tool: {name}"}
    try:
        return fn(image_bytes, **(arguments or {}))
    except Exception as exc:  # keep the agent loop alive
        return {"error": f"{type(exc).__name__}: {exc}"}
