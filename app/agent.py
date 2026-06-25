"""Agentic insurance-claim assessment workflow, fully instrumented for Datadog
LLM Observability.

Given a photo of damage (e.g. a vehicle), the agent inspects the image,
classifies the damage, applies the policy rules via the `assess_claim` tool to
decide APPROVE / MANUAL_REVIEW / REJECT, and saves a claim report.

Span tree produced per run (visible in the LLM Observability trace explorer):

    insurance_claim_agent           (workflow)
    ├─ inspect_image                (tool)   - get_image_properties
    ├─ reasoning_step               (task)
    │   └─ Bedrock Converse         (llm)    - auto-instrumented
    ├─ assess_claim                 (tool)   - policy decision
    ├─ save_report                  (tool)
    └─ ...                          (task/llm/tool repeated until end_turn)
"""
import time

from botocore.exceptions import ClientError
from ddtrace.llmobs import LLMObs
from ddtrace.llmobs.decorators import task, tool, workflow

from app import tools as toolkit
from app.bedrock import (
    MAX_RETRIES_PER_MODEL,
    _FORMAT_BY_MEDIA_TYPE,
    _RETRYABLE,
    _model_order,
    get_client,
)

MAX_AGENT_STEPS = 4


# ── Tools (each call shows up as a `tool` span) ─────────────────────────────
@tool(name="inspect_image")
def _tool_image_properties(image_bytes, **kwargs):
    result = toolkit.get_image_properties(image_bytes, **kwargs)
    LLMObs.annotate(input_data=kwargs or {"top_colors": 5}, output_data=result)
    return result


@tool(name="assess_claim")
def _tool_assess_claim(image_bytes, **kwargs):
    result = toolkit.assess_claim(image_bytes, **kwargs)
    LLMObs.annotate(input_data=kwargs, output_data=result)
    return result


@tool(name="save_report")
def _tool_save_report(image_bytes, **kwargs):
    result = toolkit.save_report(image_bytes, **kwargs)
    LLMObs.annotate(input_data=kwargs, output_data=result)
    return result


_TOOL_FUNCS = {
    "get_image_properties": _tool_image_properties,
    "assess_claim": _tool_assess_claim,
    "save_report": _tool_save_report,
}


# ── LLM step (Bedrock Converse, resilient across flapping model access) ─────
@task(name="reasoning_step")
def _converse(messages, tool_config):
    last_error = None
    for model_id in _model_order():
        for attempt in range(MAX_RETRIES_PER_MODEL + 1):
            try:
                response = get_client().converse(
                    modelId=model_id,
                    messages=messages,
                    toolConfig=tool_config,
                    inferenceConfig={"maxTokens": 1024, "temperature": 0.2},
                )
                return response, model_id
            except ClientError as exc:
                last_error = exc
                code = exc.response.get("Error", {}).get("Code", "")
                if code not in _RETRYABLE:
                    raise
                if attempt < MAX_RETRIES_PER_MODEL:
                    time.sleep(0.5 * (attempt + 1))
    raise last_error


# ── Workflow (the whole agent run = one trace) ──────────────────────────────
@workflow(name="insurance_claim_agent")
def run_agent(image_bytes: bytes, media_type: str, prompt: str) -> dict:
    image_format = _FORMAT_BY_MEDIA_TYPE.get(media_type)
    if image_format is None:
        raise ValueError(f"Unsupported media type: {media_type}")

    LLMObs.annotate(input_data=prompt)

    # Deterministic first tool call: gives the model exact facts and guarantees
    # a `tool` span in every trace.
    props = _TOOL_FUNCS["get_image_properties"](image_bytes, top_colors=5)
    tool_calls = [{"tool": "inspect_image", "result": props}]

    system_prompt = (
        "You are an auto-insurance claims assessor. Inspect the damage photo and: "
        "(1) identify the damage_type, (2) rate severity as none/minor/moderate/"
        "severe, (3) estimate repair cost in USD. Then call the assess_claim tool "
        "with those values to get the policy decision (APPROVE / MANUAL_REVIEW / "
        "REJECT). Finally call save_report with a concise claim summary including "
        "the decision and payout. "
        f"Exact image properties already extracted: {props}."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"image": {"format": image_format, "source": {"bytes": image_bytes}}},
                {"text": f"{system_prompt}\n\nClaim context: {prompt}"},
            ],
        }
    ]
    tool_config = {"tools": toolkit.TOOL_SPECS}

    final_text = ""
    model_used = None
    for _ in range(MAX_AGENT_STEPS):
        response, model_used = _converse(messages, tool_config)
        out_msg = response["output"]["message"]
        messages.append(out_msg)

        if response.get("stopReason") == "tool_use":
            tool_results = []
            for block in out_msg["content"]:
                tu = block.get("toolUse")
                if not tu:
                    continue
                fn = _TOOL_FUNCS.get(tu["name"])
                result = (
                    fn(image_bytes, **(tu.get("input") or {}))
                    if fn
                    else {"error": f"unknown tool {tu['name']}"}
                )
                tool_calls.append({"tool": tu["name"], "result": result})
                tool_results.append(
                    {
                        "toolResult": {
                            "toolUseId": tu["toolUseId"],
                            "content": [{"json": result}],
                        }
                    }
                )
            messages.append({"role": "user", "content": tool_results})
            continue

        final_text = "".join(
            b.get("text", "") for b in out_msg["content"] if "text" in b
        )
        break

    # Surface the structured policy decision from the assess_claim tool call.
    decision = next(
        (tc["result"] for tc in tool_calls if tc["tool"] == "assess_claim"),
        None,
    )

    out = {
        "answer": final_text,
        "decision": decision,
        "tool_calls": tool_calls,
        "model": model_used,
    }
    LLMObs.annotate(output_data=final_text or out)
    return out
