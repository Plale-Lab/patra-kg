#!/usr/bin/env python3
"""M4: Patra → MLHub Augmentation (Step 2 of the HF → Patra → MLHub pipeline).

Takes a Patra card already augmented by step 1 (M3 chain-of-thought) plus the
Patra catalog's experiments[] evidence (mock fixture), and emits an MLHub
ModelMetadata record wrapped in a ListModelsResponse envelope.

Pipeline:
  Phase A  Direct copy from Patra   — 10 fields, confidence 1.00
  Phase B  Compute from experiments  — 6 fields, confidence 0.95 / 0.75 / 0.00
  Phase C  One LLM call              — 20 fields, confidence 0.85 / 0.60 / 0.40
  Phase D  Envelope assembly         — ListModelsResponse wrapping result[]

Usage:
    python poc/augment_mlhub.py --card-id 1
    python poc/augment_mlhub.py               # process all model cards

Outputs:
    poc/results_m4_patra_to_mlhub.json   — CardResult[] (for judge)
    poc/mlhub_response.json              — ListModelsResponse envelope
    poc/metrics_comparison.csv           — appends M4 rows
"""

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from dotenv import load_dotenv
import httpx

sys.path.insert(0, str(Path(__file__).parent))
import runtime_formulas

load_dotenv(Path(__file__).parent.parent / ".env")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LITELLM_BASE = "https://litellm.pods.tacc.tapis.io"
TAPIS_AUTH_URL = "https://tacc.tapis.io/v3/oauth2/tokens"
GENERATOR_MODEL = "llama4-17b"
PHASE_LABEL = "M4: Patra→MLHub"

POC_DIR = Path(__file__).parent
M3_RESULTS_PATH = POC_DIR / "results_m3_chain-of-thought.json"
EXPERIMENTS_PATH = POC_DIR / "mock_experiments.json"
HF_CARDS_PATH = POC_DIR / "real_hf_cards.json"
RESULTS_OUT = POC_DIR / "results_m4_patra_to_mlhub.json"
MLHUB_RESPONSE_OUT = POC_DIR / "mlhub_response.json"
AUGMENTED_MLHUB_DIR = POC_DIR / "augmented_mlhub_cards"
METRICS_CSV = POC_DIR / "metrics_comparison.csv"

# ---------------------------------------------------------------------------
# MLHub schema constants
# ---------------------------------------------------------------------------

# Fields judged "required" for a usable MLHub card (schema has no required array)
MLHUB_REQUIRED = {
    "name", "author", "task_types", "license",
    "inference_hardware", "inference_max_latency_ms",
}

# Hand-built (patra.category, patra.input_type) → MLHub Task enum lookup.
# Misses log to stderr so the dict can grow organically.
TASK_TYPES_LOOKUP = {
    ("classification", "Image"): ["ImageClassification"],
    ("classification", "Text"): ["TextClassification"],
    ("classification", "Audio"): ["AudioClassification"],
    ("classification", "Tabular"): ["TabularClassification"],
    ("computer vision", "Image"): ["ImageClassification", "ObjectDetection"],
    ("computer vision", "Video"): ["VideoClassification"],
    ("object detection", "Image"): ["ObjectDetection"],
    ("natural language processing", "Text"): ["TextClassification", "FillMask", "TextGeneration"],
    ("natural language processing", "Multimodal"): ["ImageTextToText", "VisualQuestionAnswering"],
    ("natural language processing", "Audio"): ["AutomaticSpeechRecognition"],
    ("embedding learning", "Text"): ["SentenceSimilarity", "FeatureExtraction"],
    ("embedding learning", "Image"): ["ImageFeatureExtraction"],
    ("generative modeling", "Text"): ["TextGeneration"],
    ("generative modeling", "Image"): ["TextToImage", "UnconditionalImageGeneration"],
    ("sequence modeling", "Text"): ["FillMask", "TextGeneration"],
    ("self-supervised learning", "Text"): ["FillMask", "FeatureExtraction"],
    ("self-supervised learning", "Image"): ["ImageFeatureExtraction"],
    ("transfer learning", "Text"): ["FillMask", "FeatureExtraction"],
    ("transfer learning", "Image"): ["ImageClassification"],
}

# MLHub ModelMetadata: 36 fields. Grouped by which phase fills them.
MLHUB_ALL_FIELDS = [
    # Category A — direct copy (9)
    "name", "author", "model_type", "license", "task_types",
    "inference_software_dependencies", "pretraining_datasets", "keywords", "libraries",
    # Category B — formula from experiments[] (6)
    "inference_hardware", "inference_max_energy_consumption_watts",
    "inference_max_latency_ms", "inference_min_throughput",
    "inference_max_memory_usage_mb", "inference_distributed",
    # Category C — LLM-inferred (21)
    "image", "multi_modal", "model_inputs", "model_outputs",
    "inference_precision", "inference_max_compute_utilization_percentage",
    "pretrained", "edge_optimized", "quantization_aware", "supports_quantization",
    "pruned", "slimmed",
    "training_time", "training_precision", "training_hardware",
    "training_max_energy_consumption_watts", "training_distributed",
    "finetuning_datasets", "regulatory", "bias_evaluation_score", "annotations",
]
assert len(MLHUB_ALL_FIELDS) == 36, f"expected 36 MLHub fields, got {len(MLHUB_ALL_FIELDS)}"

CATEGORY_A_FIELDS = set(MLHUB_ALL_FIELDS[:9])
CATEGORY_B_FIELDS = set(MLHUB_ALL_FIELDS[9:15])
CATEGORY_C_FIELDS = set(MLHUB_ALL_FIELDS[15:])

LLM_FIELD_DESCRIPTIONS = {
    "image": "URL of a representative image / thumbnail for the model card.",
    "multi_modal": "Boolean — accepts multiple input modalities (text+image, etc.).",
    "model_inputs": "Array of {data_type, shape} describing each input tensor.",
    "model_outputs": "Array of {data_type, shape} describing each output tensor.",
    "inference_precision": "Numeric precision at inference, e.g. fp32, fp16, bf16, int8.",
    "inference_max_compute_utilization_percentage": "Integer 0–100; peak compute utilization.",
    "pretrained": "Boolean — released as a pretrained checkpoint.",
    "edge_optimized": "Boolean — designed or tuned for edge deployment.",
    "quantization_aware": "Boolean — trained with quantization-aware training.",
    "supports_quantization": "Boolean — can be post-training quantized.",
    "pruned": "Boolean — weights have been pruned.",
    "slimmed": "Boolean — distilled or otherwise size-reduced.",
    "training_time": "Integer hours spent training.",
    "training_precision": "Numeric precision during training, e.g. fp16, bf16.",
    "training_hardware": "HardwareRequirements object for training.",
    "training_max_energy_consumption_watts": "Integer peak watts during training.",
    "training_distributed": "Boolean — trained across multiple devices.",
    "finetuning_datasets": "Array of dataset names used for fine-tuning (distinct from pretraining).",
    "regulatory": "Array of compliance tags (e.g. HIPAA, GDPR), [] if none.",
    "bias_evaluation_score": "Integer 0–100; higher = less bias. Reduce from patra.bias_analysis if present.",
    "annotations": "Free-form object with provenance notes.",
}

# ---------------------------------------------------------------------------
# Data classes (mirror augment_poc_v2 for judge + metrics compatibility)
# ---------------------------------------------------------------------------

@dataclass
class FieldResult:
    field_name: str
    ground_truth: str | None
    augmented_value: str | None
    confidence: float
    heuristic_score: float
    composite_score: float
    exact_match: bool
    semantic_overlap: float
    reasoning: str
    extraction_method: str
    source_score: float = 0.0
    attribute_confidence: float = 0.0


@dataclass
class CardResult:
    card_id: int
    repo_id: str
    asset_type: str
    domain: str
    model_used: str
    tier1_filled: int = 0
    tier2_filled: int = 0
    fields: list[FieldResult] = field(default_factory=list)
    llm_latency_ms: int = 0
    error: str | None = None
    completeness: float = 0.0
    sufficiency: float = 0.0
    overall_confidence: float = 0.0


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def fetch_tapis_token() -> str:
    username = os.getenv("TAPIS_USERNAME", "").strip()
    password = os.getenv("TAPIS_PASSWORD", "").strip()
    if not username or not password:
        print("ERROR: Set TAPIS_USERNAME and TAPIS_PASSWORD in poc/.env", file=sys.stderr)
        sys.exit(1)
    resp = httpx.post(
        TAPIS_AUTH_URL,
        json={"username": username, "password": password, "grant_type": "password"},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["result"]["access_token"]["access_token"]


# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------

def load_patra_card(card_id: int | None, src: Path = M3_RESULTS_PATH) -> list[dict]:
    """Load CardResult entries from M3 output; filter to model_card asset_type."""
    raw = json.loads(src.read_text())
    cards = [c for c in raw if c.get("asset_type") == "model_card" and c.get("error") is None]
    if card_id is not None:
        cards = [c for c in cards if c["card_id"] == card_id]
    return cards


def load_experiments_index() -> tuple[dict[str, list[dict]], dict[int, dict]]:
    """Return (experiments_by_repo_id, edge_devices_by_id)."""
    fixture = json.loads(EXPERIMENTS_PATH.read_text())
    devices_by_id = {d["id"]: d for d in fixture["edge_devices"]}
    by_repo: dict[str, list[dict]] = {}
    for exp in fixture["experiments"]:
        by_repo.setdefault(exp["model_id_ckn"], []).append(exp)
    return by_repo, devices_by_id


def load_hf_entry_by_card_id() -> dict[int, dict]:
    raw = json.loads(HF_CARDS_PATH.read_text())
    return {c["id"]: c for c in raw}


def patra_card_as_dict(card: dict) -> dict:
    """Flatten M3's CardResult.fields[] back into a Patra-card dict of {field_name: value}."""
    return {f["field_name"]: f["augmented_value"] for f in card.get("fields", [])
            if f.get("augmented_value") not in (None, "")}


# ---------------------------------------------------------------------------
# Phase A — direct copy
# ---------------------------------------------------------------------------

def extract_direct_copy(patra: dict) -> dict:
    """Return {MLHub_field: value_or_None} for Category A."""
    def _split_csv(v):
        if not v:
            return None
        return [s.strip() for s in str(v).split(",") if s.strip()]

    return {
        "name": patra.get("ai_model_name") or patra.get("name"),
        "author": patra.get("ai_model_owner") or patra.get("author"),
        "model_type": patra.get("ai_model_model_type"),
        "license": patra.get("ai_model_license"),
        # task_types handled separately via lookup
        "inference_software_dependencies": None,  # not in M3 targets; LLM can attempt later
        "pretraining_datasets": [patra["input_data"]] if patra.get("input_data") else None,
        "keywords": _split_csv(patra.get("keywords")),
        "libraries": [patra["ai_model_framework"]] if patra.get("ai_model_framework") else None,
    }


def task_types_from_lookup(patra: dict) -> tuple[list[str] | None, bool]:
    """Return (task_types, hit) where hit=True if the lookup matched."""
    key = (patra.get("category"), patra.get("input_type"))
    hit = key in TASK_TYPES_LOOKUP
    if not hit:
        print(f"  task_types MISS: ({key[0]!r}, {key[1]!r})", file=sys.stderr)
    return TASK_TYPES_LOOKUP.get(key), hit


# ---------------------------------------------------------------------------
# Phase B — compute from experiments[]
# ---------------------------------------------------------------------------

def _device_memory_gb(device_row: dict | None) -> int | None:
    if not device_row or device_row.get("ram_mb") is None:
        return None
    return int(round(device_row["ram_mb"] / 1024))


def build_hardware_requirements(exps: list[dict], devices_by_id: dict) -> dict | None:
    dom = runtime_formulas.dominant_device(exps, devices_by_id)
    if not dom:
        return None
    mem_gb = _device_memory_gb(dom)
    accelerators = None
    if dom.get("gpu"):
        accelerators = [{
            "accelerator_type": dom["device_type"],
            "cores": None,
            "memory_gb": mem_gb,
            "system_requirements": [],
        }]
    return {
        "accelerators": accelerators,
        "architectures": None,
        "cpus": None,
        "disk_gb": None,
        "memory_gb": mem_gb,
    }


def compute_phase_b(exps: list[dict], devices_by_id: dict) -> dict:
    """Return {MLHub_field: value_or_None} for Category B."""
    if not exps:
        return {f: None for f in CATEGORY_B_FIELDS}
    p95_lat = runtime_formulas.p95_latency_ms(exps)
    p95_pow = runtime_formulas.p95_total_power_w(exps)
    dom = runtime_formulas.dominant_device(exps, devices_by_id)
    return {
        "inference_hardware": build_hardware_requirements(exps, devices_by_id),
        "inference_max_energy_consumption_watts":
            int(round(p95_pow)) if p95_pow is not None else None,
        "inference_max_latency_ms":
            int(round(p95_lat)) if p95_lat is not None else None,
        "inference_min_throughput":
            runtime_formulas.min_throughput(exps),
        "inference_max_memory_usage_mb":
            dom["ram_mb"] if dom and dom.get("ram_mb") is not None else None,
        "inference_distributed": runtime_formulas.any_distributed(exps),
    }


def formula_confidence(n_refs: int) -> float:
    if n_refs >= 3:
        return 0.95
    if n_refs >= 1:
        return 0.75
    return 0.0


# ---------------------------------------------------------------------------
# Phase C — one LLM call for 20 fields
# ---------------------------------------------------------------------------

LLM_PROMPT = """You are mapping a Patra model card into MLHub's ModelMetadata schema.

Use the Patra card below as the authoritative source of truth. Fill in the fields
listed under "Fields to fill" using values you can ground in the Patra card,
README, or experiments sample. If there is no source support, return the best
guess WITH source_strength="none" so the downstream judge can reveal it honestly.

## Patra card (already augmented)
{patra_card_json}

## README excerpt
{readme_body}

## Experiments sample (recent runs on edge devices)
{exps_summary}

## Fields to fill
{fields_spec_json}

For each field, answer:
  value:            the MLHub-valid value (null if truly unknown)
  source_strength:  "strong" — value is explicit in Patra or README
                    "weak"   — value is a reasonable inference from evidence
                    "none"   — value is a guess with no source support
  reasoning:        one sentence

Return ONLY valid JSON with this exact shape (no prose, no markdown fences):
{{
  "<field_name>": {{"value": <value>, "source_strength": "<tier>", "reasoning": "<text>"}},
  ...
}}
"""


def summarize_experiments_for_prompt(exps: list[dict], devices_by_id: dict) -> str:
    if not exps:
        return "(no matching experiments)"
    lines = []
    for e in exps[:3]:
        dev = devices_by_id.get(e["edge_device_id"], {}).get("device_type", "?")
        cpu_w = e.get("total_cpu_power_w", 0)
        gpu_w = e.get("total_gpu_power_w", 0)
        f1 = e.get("f1_score")
        lat = e.get("per_image_latency_ms")
        lines.append(
            f"- device={dev} images={e.get('total_images')} "
            f"f1={f1} latency_ms={lat} power_w={cpu_w + gpu_w:.1f}"
        )
    if len(exps) > 3:
        lines.append(f"  ... and {len(exps) - 3} more experiments")
    return "\n".join(lines)


def call_llm_mlhub(token: str, patra_card_json: str, readme_body: str,
                   exps_summary: str) -> tuple[dict, int]:
    fields_spec = {f: LLM_FIELD_DESCRIPTIONS[f] for f in CATEGORY_C_FIELDS}
    prompt = LLM_PROMPT.format(
        patra_card_json=patra_card_json,
        readme_body=(readme_body or "(no README available)")[:2000],
        exps_summary=exps_summary,
        fields_spec_json=json.dumps(fields_spec, indent=2),
    )
    t0 = time.time()
    resp = httpx.post(
        f"{LITELLM_BASE}/v1/chat/completions",
        headers={"Content-Type": "application/json", "X-Tapis-Token": token},
        json={
            "model": GENERATOR_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 2000,
        },
        timeout=120,
    )
    latency_ms = int((time.time() - t0) * 1000)
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"].strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw), latency_ms


SOURCE_STRENGTH_TO_CONFIDENCE = {"strong": 0.85, "weak": 0.60, "none": 0.40}


# ---------------------------------------------------------------------------
# Envelope + metrics
# ---------------------------------------------------------------------------

def build_list_models_response(mm_objects: list[dict]) -> dict:
    return {
        "result": mm_objects,
        "status": 200,
        "message": f"Augmented {len(mm_objects)} card(s) from Patra via {PHASE_LABEL}",
        "metadata": {
            "source": "patra",
            "pipeline": PHASE_LABEL,
            "n_cards": len(mm_objects),
        },
        "version": "0.1.0",
    }


def append_metrics_csv(all_results: list[CardResult], phase: str) -> None:
    write_header = not METRICS_CSV.exists()
    # Bucket FieldResults by field_name across all cards
    by_field: dict[str, list[FieldResult]] = {}
    for r in all_results:
        for f in r.fields:
            by_field.setdefault(f.field_name, []).append(f)

    with open(METRICS_CSV, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(["phase", "asset_type", "field", "extraction_method",
                             "n", "attr_confidence", "composite",
                             "exact_pct", "semantic_overlap", "coverage_pct"])
        for fname in MLHUB_ALL_FIELDS:
            flds = by_field.get(fname, [])
            if not flds:
                continue
            n = len(flds)
            methods = {f.extraction_method for f in flds}
            method = methods.pop() if len(methods) == 1 else "mixed"
            avg_ac = sum(f.attribute_confidence for f in flds) / n
            avg_comp = sum(f.composite_score for f in flds) / n
            exact_pct = 100 * sum(1 for f in flds if f.exact_match) / n
            avg_overlap = sum(f.semantic_overlap for f in flds) / n
            cover_pct = 100 * sum(1 for f in flds if f.augmented_value) / n
            writer.writerow([
                phase, "model_card", fname, method, n,
                f"{avg_ac:.3f}", f"{avg_comp:.3f}",
                f"{exact_pct:.1f}", f"{avg_overlap:.2f}", f"{cover_pct:.1f}",
            ])


# ---------------------------------------------------------------------------
# Per-card runner
# ---------------------------------------------------------------------------

def _stringify(val) -> str | None:
    """Convert an MLHub value into the string representation used in CardResult."""
    if val is None:
        return None
    if isinstance(val, (str,)):
        return val
    return json.dumps(val, default=str)


def augment_one_card(card: dict, experiments_by_repo: dict, devices_by_id: dict,
                     hf_by_id: dict, token: str) -> tuple[CardResult, dict]:
    card_id = card["card_id"]
    repo_id = card["repo_id"]
    patra = patra_card_as_dict(card)
    exps = experiments_by_repo.get(repo_id, [])
    print(f"\n[card #{card_id}] repo={repo_id}  exact_match={'Y' if exps else 'N'}  "
          f"n_exps={len(exps)}", flush=True)

    # Phase A — direct copy
    a_values = extract_direct_copy(patra)
    tt_value, tt_hit = task_types_from_lookup(patra)
    a_values["task_types"] = tt_value

    # Phase B — formulas
    b_values = compute_phase_b(exps, devices_by_id)
    b_conf = formula_confidence(len(exps))

    # Phase C — LLM
    hf_entry = hf_by_id.get(card_id, {})
    readme = hf_entry.get("readme_body", "")
    patra_json_str = json.dumps(patra, indent=2, default=str)[:4000]
    exps_summary = summarize_experiments_for_prompt(exps, devices_by_id)
    try:
        c_raw, llm_latency = call_llm_mlhub(token, patra_json_str, readme, exps_summary)
    except Exception as ex:
        print(f"  LLM error: {ex}", file=sys.stderr)
        c_raw = {}
        llm_latency = 0

    # Build the MLHub ModelMetadata dict and per-field CardResult entries
    mm: dict = {}
    fields: list[FieldResult] = []
    tier1 = tier2 = 0

    # Category A
    for fname, val in a_values.items():
        mm[fname] = val
        conf = 1.00 if val is not None else 0.0
        if fname == "task_types":
            conf = 1.00 if tt_hit else 0.0
        method = "patra_direct" if fname != "task_types" else "patra_lookup"
        fields.append(FieldResult(
            field_name=fname,
            ground_truth=None,
            augmented_value=_stringify(val),
            confidence=conf,
            heuristic_score=conf,
            composite_score=conf,
            exact_match=False,
            semantic_overlap=0.0,
            reasoning="direct copy from Patra" if fname != "task_types" else
                      ("task_types from (category, input_type) lookup" if tt_hit else
                       "task_types lookup miss"),
            extraction_method=method,
            source_score=conf,
            attribute_confidence=conf,
        ))
        if val is not None:
            tier1 += 1

    # Category B
    for fname, val in b_values.items():
        mm[fname] = val
        conf = b_conf if val is not None else 0.0
        fields.append(FieldResult(
            field_name=fname,
            ground_truth=None,
            augmented_value=_stringify(val),
            confidence=conf,
            heuristic_score=conf,
            composite_score=conf,
            exact_match=False,
            semantic_overlap=0.0,
            reasoning=f"aggregated over {len(exps)} experiments" if exps
                      else "cold-start: no matching experiments",
            extraction_method="experiment_formula",
            source_score=conf,
            attribute_confidence=conf,
        ))
        if val is not None:
            tier1 += 1

    # Category C
    for fname in CATEGORY_C_FIELDS:
        entry = c_raw.get(fname) if isinstance(c_raw, dict) else None
        if not isinstance(entry, dict):
            mm[fname] = None
            fields.append(FieldResult(
                field_name=fname,
                ground_truth=None,
                augmented_value=None,
                confidence=0.0,
                heuristic_score=0.0,
                composite_score=0.0,
                exact_match=False,
                semantic_overlap=0.0,
                reasoning="LLM did not return this field",
                extraction_method="patra_llm",
                source_score=0.0,
                attribute_confidence=0.0,
            ))
            continue
        val = entry.get("value")
        strength = entry.get("source_strength", "none")
        conf = SOURCE_STRENGTH_TO_CONFIDENCE.get(strength, 0.40)
        if val is None:
            conf = 0.0
        mm[fname] = val
        fields.append(FieldResult(
            field_name=fname,
            ground_truth=None,
            augmented_value=_stringify(val),
            confidence=conf,
            heuristic_score=conf,
            composite_score=conf,
            exact_match=False,
            semantic_overlap=0.0,
            reasoning=entry.get("reasoning", "")[:160],
            extraction_method="patra_llm",
            source_score=conf,
            attribute_confidence=conf,
        ))
        if val is not None:
            tier2 += 1

    # Summary stats
    filled = sum(1 for f in fields if f.augmented_value)
    n_required_filled = sum(1 for f in fields
                            if f.field_name in MLHUB_REQUIRED and f.augmented_value)
    completeness = filled / len(MLHUB_ALL_FIELDS)
    sufficiency = n_required_filled / len(MLHUB_REQUIRED)
    overall_conf = sum(f.attribute_confidence for f in fields) / len(fields) if fields else 0.0

    print(f"  filled={filled}/{len(MLHUB_ALL_FIELDS)}  "
          f"required={n_required_filled}/{len(MLHUB_REQUIRED)}  "
          f"overall_conf={overall_conf:.3f}  llm_ms={llm_latency}")

    result = CardResult(
        card_id=card_id,
        repo_id=repo_id,
        asset_type="model_card",
        domain=card.get("domain", ""),
        model_used=GENERATOR_MODEL,
        tier1_filled=tier1,
        tier2_filled=tier2,
        fields=fields,
        llm_latency_ms=llm_latency,
        completeness=completeness,
        sufficiency=sufficiency,
        overall_confidence=overall_conf,
    )
    return result, mm


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="M4: Patra → MLHub Augmentation")
    parser.add_argument("--card-id", type=int, default=None,
                        help="Process only this card_id; default processes all model cards.")
    parser.add_argument("--input", type=Path, default=M3_RESULTS_PATH,
                        help="M3 results JSON to use as Patra card source.")
    args = parser.parse_args()

    cards = load_patra_card(args.card_id, src=args.input)
    if not cards:
        print(f"ERROR: no cards found (card_id={args.card_id})", file=sys.stderr)
        return 1

    print(f"Loading fixtures...")
    experiments_by_repo, devices_by_id = load_experiments_index()
    hf_by_id = load_hf_entry_by_card_id()
    print(f"  {len(experiments_by_repo)} repos with experiments, "
          f"{len(devices_by_id)} edge devices, {len(hf_by_id)} HF cards.")

    print(f"Fetching Tapis token...")
    token = fetch_tapis_token()
    print(f"  token acquired.\n")

    results: list[CardResult] = []
    mm_objects: list[dict] = []
    AUGMENTED_MLHUB_DIR.mkdir(exist_ok=True)
    for card in cards:
        res, mm = augment_one_card(card, experiments_by_repo, devices_by_id, hf_by_id, token)
        results.append(res)
        mm_objects.append(mm)

        # Per-card file (mirrors poc/augmented_cards/ naming from step 1)
        safe_repo = res.repo_id.replace("/", "_").lower()
        per_card_path = AUGMENTED_MLHUB_DIR / f"{res.card_id:02d}_{safe_repo}.json"
        per_card_path.write_text(json.dumps(mm, indent=2, default=str))

        time.sleep(0.3)  # be gentle with the LLM endpoint

    # Write outputs
    RESULTS_OUT.write_text(json.dumps([asdict(r) for r in results], indent=2, default=str))
    MLHUB_RESPONSE_OUT.write_text(json.dumps(build_list_models_response(mm_objects),
                                             indent=2, default=str))
    append_metrics_csv(results, phase=PHASE_LABEL)

    print(f"\n=== M4 Summary ===")
    print(f"  Cards processed:   {len(results)}")
    print(f"  Avg completeness:  {sum(r.completeness for r in results) / len(results):.3f}")
    print(f"  Avg sufficiency:   {sum(r.sufficiency for r in results) / len(results):.3f}")
    print(f"  Avg confidence:    {sum(r.overall_confidence for r in results) / len(results):.3f}")
    print(f"  Output files:")
    print(f"    {RESULTS_OUT}")
    print(f"    {MLHUB_RESPONSE_OUT}")
    print(f"    {AUGMENTED_MLHUB_DIR}/  ({len(results)} per-card files)")
    print(f"    {METRICS_CSV}  (appended)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
