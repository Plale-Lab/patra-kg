"""M4: Experiment-Grounded Runtime Augmentation.

Compute-then-LLM pipeline that fills runtime model-card fields by querying
historical experiments for similar models in the Patra catalog (mocked via
patra_query). The method is called "experiment-grounded" because every value
is either computed from or summarized over retrieved experiment rows.

Note on naming: the query layer's function signatures mirror the tools exposed
by the existing MCP server (mcp_server/main.py) so the production path can
swap direct reads for MCP tool calls with no caller changes. The POC itself
does not speak MCP — the grounding is on experiment evidence, not on MCP
as a transport.

6 objective fields are computed deterministically from retrieved evidence.
2 free-text fields (typical_deployment_context, known_failure_modes) are
written by a single LLM call grounded on the computed summary + low-score
image sample. No LLM call when retrieval is empty.

Confidence rubric (split by source — formulas can't drift from their evidence;
LLM freetext can, so it gets a lower cap):
    formula-computed, >= 3 refs -> 0.95
    formula-computed, 1-2 refs  -> 0.75
    LLM freetext,     >= 3 refs -> 0.85
    LLM freetext,     1-2 refs  -> 0.60
    0 refs / insufficient       -> 0.00
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from patra_query import (
    get_experiment_images,
    get_experiments_for_models,
    search_similar_models,
)
import runtime_formulas

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

MC_RUNTIME_FIELDS: dict[str, str] = {
    "runtime_suggested_hardware": "Most likely edge_devices.device_type for deployment (e.g. 'Jetson Nano', 'A100', 'CPU-x86'). Argmax over similar-model experiment device counts.",
    "runtime_expected_f1_range": "[p25, p75] of f1_score across similar-model experiments.",
    "runtime_expected_latency_ms": "Median per-image latency in ms across similar-model experiments.",
    "runtime_typical_deployment_context": "One-sentence description (<=200 chars) of typical deployment context, grounded in the computed summary.",
    "runtime_deployment_maturity": "One of: experimental | validated | production — derived from distinct-devices and run count among similar-model experiments.",
    "runtime_recommended_min_ram_mb": "Minimum RAM (MB) from edge_devices where similar models achieved f1 >= 0.6.",
    "runtime_inference_cost_class": "One of: low | medium | high — bucketed from median latency and dominant device class.",
    "runtime_expected_total_power_w": "Median total power draw (total_cpu_power_w + total_gpu_power_w, in watts) across similar-model experiments.",
    "runtime_known_failure_modes": "One-sentence description (<=200 chars) of observed failure patterns in low-score experiment_images, grounded in evidence. Empty string if no low-score evidence.",
}

RUNTIME_DETERMINISTIC_FIELDS = {
    "runtime_suggested_hardware",
    "runtime_expected_f1_range",
    "runtime_expected_latency_ms",
    "runtime_deployment_maturity",
    "runtime_recommended_min_ram_mb",
    "runtime_inference_cost_class",
    "runtime_expected_total_power_w",
}
RUNTIME_LLM_FIELDS = {
    "runtime_typical_deployment_context",
    "runtime_known_failure_modes",
}

# ---------------------------------------------------------------------------
# Device metadata (mirrors edge_devices table; loaded from the fixture)
# ---------------------------------------------------------------------------

# Populated lazily from the fixture by _ensure_device_metadata().
_DEVICES_BY_ID: dict[int, dict] = {}
_VALID_DEVICE_TYPES: set[str] = set()


def _ensure_device_metadata() -> None:
    """Load device metadata from the fixture on first use."""
    global _DEVICES_BY_ID, _VALID_DEVICE_TYPES
    if _DEVICES_BY_ID:
        return
    from patra_query import _load  # noqa: WPS437 — test seam
    fixture = _load()
    _DEVICES_BY_ID = {d["id"]: d for d in fixture["edge_devices"]}
    _VALID_DEVICE_TYPES = {d["device_type"] for d in fixture["edge_devices"]}


# ---------------------------------------------------------------------------
# Aggregation adapters — thin wrappers that load device metadata then delegate
# to runtime_formulas. Ground-truth builder uses the same formulas directly.
# ---------------------------------------------------------------------------

def _suggested_hardware(exps: list[dict]) -> str | None:
    _ensure_device_metadata()
    return runtime_formulas.suggested_hardware(exps, _DEVICES_BY_ID, _VALID_DEVICE_TYPES)


def _latency_median_ms(exps: list[dict]) -> float | None:
    return runtime_formulas.expected_latency_ms(exps)


def _deployment_maturity(exps: list[dict]) -> str | None:
    return runtime_formulas.deployment_maturity(exps)


def _inference_cost_class(exps: list[dict]) -> str | None:
    _ensure_device_metadata()
    return runtime_formulas.inference_cost_class(exps, _DEVICES_BY_ID)


def _min_ram_mb(exps: list[dict]) -> int | None:
    _ensure_device_metadata()
    return runtime_formulas.recommended_min_ram_mb(exps, _DEVICES_BY_ID)


def _expected_f1_range(exps: list[dict]) -> list[float] | None:
    return runtime_formulas.expected_f1_range(exps)


def _expected_total_power_w(exps: list[dict]) -> float | None:
    return runtime_formulas.expected_total_power_w(exps)


# ---------------------------------------------------------------------------
# Compute-then-LLM pipeline
# ---------------------------------------------------------------------------

@dataclass
class RuntimeResult:
    values: dict[str, object]           # field_name -> value (native types, JSON-serializable)
    confidences: dict[str, float]       # field_name -> attr_confidence
    reasoning: dict[str, str]           # field_name -> one-line explanation
    reference_experiments: list[int]    # experiment IDs used
    similar_model_ids: list[int]
    cold_start: bool                    # True: no similar models found at all
    evidence_empty: bool                # True: similar models found but had no experiments
    llm_latency_ms: int
    low_score_image_count: int


def _formula_confidence(n_refs: int) -> float:
    """Confidence for deterministically computed fields.

    Higher cap than freetext: a formula's output cannot drift from its evidence,
    so high evidence count should translate into high confidence.
    """
    if n_refs >= 3:
        return 0.95
    if n_refs >= 1:
        return 0.75
    return 0.0


def _freetext_confidence(n_refs: int) -> float:
    """Confidence for LLM-generated freetext fields.

    Capped lower than formula confidence: the LLM can paraphrase or drift
    even when the evidence is strong, so confidence should reflect that.
    """
    if n_refs >= 3:
        return 0.85
    if n_refs >= 1:
        return 0.60
    return 0.0


def _empty_runtime(cold_start: bool, evidence_empty: bool, similar_model_ids: list[int]) -> RuntimeResult:
    reason = ("Cold start — no similar models in catalog" if cold_start
              else "Similar models found but none have recorded experiments yet")
    return RuntimeResult(
        values={k: None for k in MC_RUNTIME_FIELDS},
        confidences={k: 0.0 for k in MC_RUNTIME_FIELDS},
        reasoning={k: reason for k in MC_RUNTIME_FIELDS},
        reference_experiments=[],
        similar_model_ids=similar_model_ids,
        cold_start=cold_start,
        evidence_empty=evidence_empty,
        llm_latency_ms=0,
        low_score_image_count=0,
    )


_LLM_PROMPT = """You are writing two short freetext fields for a Patra model card, grounded only on the computed deployment summary below. Do NOT invent values — only restate and summarize what is given. Keep each field under 200 characters. Return ONLY valid JSON.

## Card
{name} (category={category}, input_type={input_type})

## Computed deployment summary (authoritative)
- suggested_hardware: {suggested_hardware}
- deployment_maturity: {deployment_maturity}
- expected_f1_range: {expected_f1_range}
- expected_latency_ms: {expected_latency_ms}
- expected_total_power_w: {expected_total_power_w}
- inference_cost_class: {inference_cost_class}
- reference_experiments: {reference_experiments}

## Low-score image sample (for known_failure_modes grounding; may be empty)
{low_score_images}

## Fields to write
- typical_deployment_context: one sentence describing the typical usage pattern implied by the summary above.
- known_failure_modes: one sentence describing failure patterns visible in the low-score image sample. Empty string "" if the sample is empty.

Return ONLY this JSON object (no prose, no markdown):
{{"typical_deployment_context": "...", "known_failure_modes": "..."}}
"""


def run_experiment_augmentation(
    hf_card_id: int,
    name: str,
    category: str,
    input_type: str,
    llm_call: Callable[[str, int, float], tuple[str, int]] | None = None,
    similar_limit: int = 5,
    low_score_image_limit: int = 10,
) -> RuntimeResult:
    """Run the M4 compute-then-LLM pipeline for one model card.

    Parameters
    ----------
    llm_call : (prompt, max_tokens, temperature) -> (response_text, latency_ms)
        Caller-supplied LLM dispatch. Pass None to skip the LLM call entirely
        (the two freetext fields will be filled with null, confidence 0).
    """
    similar = search_similar_models(category, input_type, limit=similar_limit,
                                    exclude_hf_card_id=hf_card_id)
    if not similar:
        return _empty_runtime(cold_start=True, evidence_empty=False, similar_model_ids=[])

    model_ids = [s["model_id"] for s in similar]
    exps = get_experiments_for_models(model_ids)

    if not exps:
        return _empty_runtime(cold_start=False, evidence_empty=True, similar_model_ids=model_ids)

    # Phase 3a — deterministic aggregation
    values: dict[str, object] = {
        "runtime_suggested_hardware": _suggested_hardware(exps),
        "runtime_expected_f1_range": _expected_f1_range(exps),
        "runtime_expected_latency_ms": _latency_median_ms(exps),
        "runtime_deployment_maturity": _deployment_maturity(exps),
        "runtime_recommended_min_ram_mb": _min_ram_mb(exps),
        "runtime_inference_cost_class": _inference_cost_class(exps),
        "runtime_expected_total_power_w": _expected_total_power_w(exps),
    }

    ref_ids = [e["id"] for e in exps]
    n_refs = len(ref_ids)
    formula_conf = _formula_confidence(n_refs)
    freetext_conf = _freetext_confidence(n_refs)
    confidences: dict[str, float] = {k: (formula_conf if values[k] is not None else 0.0)
                                     for k in RUNTIME_DETERMINISTIC_FIELDS}
    reasoning: dict[str, str] = {
        k: f"Computed from {n_refs} similar-model experiments" for k in RUNTIME_DETERMINISTIC_FIELDS
    }

    # Phase 3b — LLM freetext
    low_score_imgs = get_experiment_images(ref_ids, only_low_score=True, limit=low_score_image_limit)
    low_score_rows = [
        {"image_name": img["image_name"], "top_label": img["top_label"],
         "top_probability": img["top_probability"],
         "ground_truth": img["ground_truth"], "experiment_id": img["experiment_id"]}
        for img in low_score_imgs
    ]

    freetext_values: dict[str, str | None] = {k: None for k in RUNTIME_LLM_FIELDS}
    llm_latency = 0

    if llm_call is not None:
        prompt = _LLM_PROMPT.format(
            name=name, category=category, input_type=input_type,
            suggested_hardware=values["runtime_suggested_hardware"],
            deployment_maturity=values["runtime_deployment_maturity"],
            expected_f1_range=values["runtime_expected_f1_range"],
            expected_latency_ms=values["runtime_expected_latency_ms"],
            expected_total_power_w=values["runtime_expected_total_power_w"],
            inference_cost_class=values["runtime_inference_cost_class"],
            reference_experiments=ref_ids[:20],  # cap to keep prompt bounded
            low_score_images=json.dumps(low_score_rows, indent=2) if low_score_rows else "(no low-score images)",
        )
        try:
            response_text, llm_latency = llm_call(prompt, 400, 0.2)
            parsed = _parse_llm_response(response_text)
            if "typical_deployment_context" in parsed:
                freetext_values["runtime_typical_deployment_context"] = parsed["typical_deployment_context"]
            if "known_failure_modes" in parsed:
                freetext_values["runtime_known_failure_modes"] = parsed["known_failure_modes"]
        except Exception as exc:
            reasoning["runtime_typical_deployment_context"] = f"LLM error: {exc}"
            reasoning["runtime_known_failure_modes"] = f"LLM error: {exc}"

    for k, v in freetext_values.items():
        values[k] = v
        # Freetext uses the freetext_conf rubric (capped lower than formula_conf).
        # known_failure_modes degrades to 0 when there's no low-score evidence
        # (the field is legitimately empty, not a failure).
        if k == "runtime_known_failure_modes" and not low_score_rows:
            confidences[k] = 0.0
            reasoning[k] = "No low-score experiment_images; field left empty"
        elif v is None:
            confidences[k] = 0.0
            reasoning.setdefault(k, "LLM call skipped or failed")
        else:
            confidences[k] = freetext_conf
            reasoning.setdefault(k, f"LLM-generated from summary, grounded in {n_refs} experiments")

    return RuntimeResult(
        values=values,
        confidences=confidences,
        reasoning=reasoning,
        reference_experiments=ref_ids,
        similar_model_ids=model_ids,
        cold_start=False,
        evidence_empty=False,
        llm_latency_ms=llm_latency,
        low_score_image_count=len(low_score_rows),
    )


def _parse_llm_response(text: str) -> dict:
    """Extract a JSON object from the LLM response (tolerant of prose / code fences)."""
    if not text:
        return {}
    # Strip <think> blocks (Qwen sometimes leaks them even with enable_thinking=False)
    import re
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Prefer fenced JSON
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidate = m.group(1) if m else text
    # Fall back to first {...} substring
    if not candidate.lstrip().startswith("{"):
        m2 = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
        if m2:
            candidate = m2.group(0)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def value_to_str(v: object) -> str | None:
    """Render a runtime value for FieldResult storage / CSV output."""
    if v is None:
        return None
    if isinstance(v, str):
        return v if v else None
    if isinstance(v, (int, float, bool)):
        return str(v)
    return json.dumps(v, separators=(",", ":"))
