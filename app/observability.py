"""Datadog LLM Observability setup (agentless).

Enabling this auto-instruments Bedrock boto3 calls so LLM traces, latency,
token usage, and errors flow to Datadog under the ml_obs.* metrics that the
dashboard widgets query. No local Datadog Agent required.
"""
import os


def _truthy(value: str) -> bool:
    return str(value).lower() in ("1", "true", "yes", "on")


def setup_llm_observability() -> bool:
    """Enable Datadog LLMObs if DD_LLMOBS_ENABLED is set. Returns True on enable."""
    if not _truthy(os.getenv("DD_LLMOBS_ENABLED", "")):
        return False

    try:
        from ddtrace.llmobs import LLMObs

        LLMObs.enable(
            ml_app=os.getenv("DD_LLMOBS_ML_APP", "image-analysis"),
            api_key=os.getenv("DD_API_KEY"),
            site=os.getenv("DD_SITE", "datadoghq.com"),
            agentless_enabled=_truthy(os.getenv("DD_LLMOBS_AGENTLESS_ENABLED", "1")),
        )
        return True
    except Exception as exc:  # never let observability break the app
        print(f"[observability] LLMObs not enabled: {exc}")
        return False
