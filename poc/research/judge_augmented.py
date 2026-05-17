#!/usr/bin/env python3
"""LLM-as-Judge: evaluate augmented field values against source material.

Uses qwen3-32b (different family from generator llama4-17b) to score
all filled fields for a card in a single API call.

Usage:
    python poc/judge_augmented.py --phase "M1: Structured Extraction" --dataset real_hf_cards.json
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
import httpx

load_dotenv(Path(__file__).parent.parent / ".env")

LITELLM_BASE = "https://litellm.pods.tacc.tapis.io"
TAPIS_AUTH_URL = "https://tacc.tapis.io/v3/oauth2/tokens"
JUDGE_MODEL = "qwen3-32b"
RESULTS_DIR = Path(__file__).parent / "data" / "outputs"
JUDGE_CSV = Path(__file__).parent / "data" / "outputs" / "judge_scores.csv"

FIELD_DESCRIPTIONS = {
    "name": "Display name for the model",
    "version": "Version string",
    "short_description": "One-sentence summary",
    "full_description": "2-4 sentence description",
    "keywords": "Comma-separated keywords",
    "author": "Creator or organization",
    "citation": "BibTeX citation or arXiv URL",
    "input_data": "Training dataset URL or name",
    "input_type": "Input data type: Image, Text, Audio, Tabular, Multimodal, Video",
    "output_data": "Description of model output",
    "foundational_model": "Base model architecture name",
    "category": "ML task category",
    "documentation": "URL for documentation",
    "is_private": "Whether the model is private",
    "is_gated": "Whether access requires approval",
    "ai_model_framework": "ML framework: sklearn, tensorflow, pytorch, other",
    "ai_model_license": "License identifier",
    "ai_model_model_type": "Architecture type: cnn, dnn, llm, gnn, etc.",
    "ai_model_version": "Model version string",
    "ai_model_description": "One-sentence description of the AI model",
    "ai_model_owner": "Owner or organization",
    "ai_model_location": "Download URL for the model",
}

BATCH_JUDGE_PROMPT = """You are a metadata quality judge for the Patra ML model catalog.

Given this HuggingFace model's API metadata and README, score EACH augmented field below.

## HuggingFace API metadata
{hf_json}

## README excerpt
{readme_excerpt}

## Fields to evaluate
{fields_json}

## Scoring rubric (apply to EACH field)
0 = wrong, misleading, or contradicted by source material
1 = acceptable but imprecise, or correct from general knowledge but not directly in source
2 = correct and directly supported by source material

Return ONLY valid JSON — an object with field names as keys:
{{
  "field_name_1": {{"score": N, "reason": "one sentence"}},
  "field_name_2": {{"score": N, "reason": "one sentence"}},
  ...
}}"""


def fetch_tapis_token() -> str:
    username = os.getenv("TAPIS_USERNAME", "").strip()
    password = os.getenv("TAPIS_PASSWORD", "").strip()
    if not username or not password:
        print("ERROR: Set TAPIS_USERNAME and TAPIS_PASSWORD in poc/.env")
        sys.exit(1)
    resp = httpx.post(TAPIS_AUTH_URL,
        json={"username": username, "password": password, "grant_type": "password"}, timeout=15)
    resp.raise_for_status()
    return resp.json()["result"]["access_token"]["access_token"]


def call_judge_batch(token: str, hf_json: str, readme: str, fields_to_judge: dict[str, str]) -> dict:
    readme_excerpt = readme[:2000] if readme else "(no README available)"
    fields_json = json.dumps(
        {fname: {"value": val, "description": FIELD_DESCRIPTIONS.get(fname, fname)}
         for fname, val in fields_to_judge.items()},
        indent=2,
    )
    prompt = BATCH_JUDGE_PROMPT.format(
        hf_json=hf_json,
        readme_excerpt=readme_excerpt,
        fields_json=fields_json,
    )
    resp = httpx.post(
        f"{LITELLM_BASE}/v1/chat/completions",
        headers={"Content-Type": "application/json", "X-Tapis-Token": token},
        json={
            "model": JUDGE_MODEL,
            "messages": [
                {"role": "system", "content": "You are a metadata quality judge. Return ONLY valid JSON. No thinking, no explanation."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 1500,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
        timeout=60,
    )
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"].strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Runtime judge (M4) — exact/tolerance match + LLM judge for 2 freetext fields
# ---------------------------------------------------------------------------

GROUND_TRUTH_PATH = Path(__file__).parent / "data" / "inputs" / "mock_experiments_ground_truth.json"

# Mapping: FieldResult.field_name (runtime_*) -> ground-truth key
GT_FIELD_MAP = {
    "runtime_suggested_hardware": "suggested_hardware",
    "runtime_expected_f1_range": "expected_f1_range",
    "runtime_expected_latency_ms": "expected_latency_ms",
    "runtime_deployment_maturity": "deployment_maturity",
    "runtime_recommended_min_ram_mb": "recommended_min_ram_mb",
    "runtime_inference_cost_class": "inference_cost_class",
    "runtime_expected_total_power_w": "expected_total_power_w",
}

RUNTIME_OBJECTIVE_FIELDS = set(GT_FIELD_MAP.keys())
RUNTIME_FREETEXT_FIELDS = {"runtime_typical_deployment_context", "runtime_known_failure_modes"}

LATENCY_TOLERANCE = 0.10  # ±10%
POWER_TOLERANCE = 0.15    # ±15% (noisier metric)

RUNTIME_JUDGE_PROMPT = """You are judging the quality of two freetext fields proposed for a Patra model card.
Both fields must be grounded in the computed deployment summary below. Invention is penalized.

## Card
{name} (category={category}, input_type={input_type})

## Authoritative computed summary (what the fields should reflect)
{summary_json}

## Low-score image sample
{low_score_images}

## Fields to evaluate
{fields_json}

## Scoring rubric per field
0 = invented content not supported by the summary, or contradicts it
1 = partially grounded; references correct items but adds unsupported claims
2 = fully grounded in the summary; no invented details

Return ONLY valid JSON:
{{
  "runtime_typical_deployment_context": {{"score": N, "reason": "one sentence"}},
  "runtime_known_failure_modes": {{"score": N, "reason": "one sentence"}}
}}"""


def _parse_runtime_value(field_name: str, raw_value: str):
    """Parse the stringified augmented_value back into a native type for comparison."""
    if raw_value is None:
        return None
    s = raw_value.strip()
    if not s:
        return None
    if field_name == "runtime_expected_f1_range":
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return None
    if field_name in ("runtime_expected_latency_ms", "runtime_expected_total_power_w"):
        try:
            return float(s)
        except ValueError:
            return None
    if field_name == "runtime_recommended_min_ram_mb":
        try:
            return int(s)
        except ValueError:
            return None
    return s


def _score_objective(field_name: str, augmented, truth) -> tuple[float, str]:
    """Return (score in [0.0, 1.0], reason). -1 if truth is null (not judged)."""
    if truth is None and augmented is None:
        return (-1.0, "both null — not judged")
    if truth is None:
        return (0.0, "fixture has no ground truth for this field")
    if augmented is None:
        return (0.0, "augmentation did not fill this field")

    if field_name == "runtime_expected_f1_range":
        # Overlap check: ranges overlap if max(lows) <= min(highs)
        if not (isinstance(augmented, list) and len(augmented) == 2 and isinstance(truth, list) and len(truth) == 2):
            return (0.0, "malformed range")
        low = max(augmented[0], truth[0])
        high = min(augmented[1], truth[1])
        return (1.0, "overlap") if low <= high else (0.0, "ranges disjoint")

    if field_name == "runtime_expected_latency_ms":
        if abs(augmented - truth) / max(truth, 1e-6) <= LATENCY_TOLERANCE:
            return (1.0, f"within ±{int(LATENCY_TOLERANCE*100)}% of {truth}")
        return (0.0, f"{augmented} vs truth {truth} — outside tolerance")

    if field_name == "runtime_expected_total_power_w":
        if abs(augmented - truth) / max(truth, 1e-6) <= POWER_TOLERANCE:
            return (1.0, f"within ±{int(POWER_TOLERANCE*100)}% of {truth}")
        return (0.0, f"{augmented} vs truth {truth} — outside tolerance")

    # Remaining: suggested_hardware, deployment_maturity, inference_cost_class, recommended_min_ram_mb -> exact
    return (1.0, "exact match") if augmented == truth else (0.0, f"{augmented!r} != {truth!r}")


def _call_runtime_freetext_judge(token: str, summary_json: str, low_score_images: str,
                                 name: str, category: str, input_type: str,
                                 fields_to_judge: dict[str, str]) -> dict:
    prompt = RUNTIME_JUDGE_PROMPT.format(
        name=name, category=category, input_type=input_type,
        summary_json=summary_json, low_score_images=low_score_images,
        fields_json=json.dumps(fields_to_judge, indent=2),
    )
    resp = httpx.post(
        f"{LITELLM_BASE}/v1/chat/completions",
        headers={"Content-Type": "application/json", "X-Tapis-Token": token},
        json={
            "model": JUDGE_MODEL,
            "messages": [
                {"role": "system", "content": "You are a metadata quality judge. Return ONLY valid JSON."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 600,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
        timeout=60,
    )
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"].strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


def run_runtime_judge(args) -> None:
    """Evaluate runtime_* fields against the fixture's ground truth + freetext judge."""
    if not GROUND_TRUTH_PATH.exists():
        print(f"ERROR: {GROUND_TRUTH_PATH} not found. Run build_mock_experiments.py first.")
        sys.exit(1)

    phase_slug = args.phase.replace(" ", "_").replace(":", "").lower()
    results_path = RESULTS_DIR / (args.results or f"results_{phase_slug}.json")
    if not results_path.exists():
        print(f"ERROR: {results_path} not found. Run augmentation first.")
        sys.exit(1)

    ground_truth = json.loads(GROUND_TRUTH_PATH.read_text())
    results = json.loads(results_path.read_text())

    # Token lazily; only needed if any card has freetext to judge
    token = None

    print(f"=== Runtime Judge ===")
    print(f"Phase:   {args.phase}")
    print(f"Results: {results_path.name}")
    print(f"Ground:  {GROUND_TRUTH_PATH.name}  ({len(ground_truth)} cards)")
    print()

    write_header = not JUDGE_CSV.exists()
    judged = 0
    score_sum = 0.0
    api_calls = 0

    with open(JUDGE_CSV, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(["phase", "card_id", "repo_id", "field", "value_excerpt",
                             "judge_score", "judge_reason", "attr_confidence"])

        for r in results:
            card_id = r["card_id"]
            repo_id = r["repo_id"]
            gt = ground_truth.get(str(card_id))

            runtime_fields = {f["field_name"]: f for f in r.get("fields", [])
                              if f["field_name"].startswith("runtime_")}
            if not runtime_fields:
                continue

            # Objective fields
            summary_for_judge = {}
            for fname in RUNTIME_OBJECTIVE_FIELDS:
                field = runtime_fields.get(fname)
                if field is None:
                    continue
                aug_raw = field.get("augmented_value")
                aug_parsed = _parse_runtime_value(fname, aug_raw)
                summary_for_judge[fname] = aug_parsed

                if gt is None or gt.get("cold_start"):
                    # Cold-start: the correct behavior is to leave runtime empty.
                    if aug_raw is None:
                        # Skip — not judged; cold-start declined correctly
                        continue
                    writer.writerow([args.phase, card_id, repo_id, fname, str(aug_raw)[:80], 0,
                                     "hallucinated under cold-start (ground truth is null)",
                                     f"{field.get('attribute_confidence', 0):.3f}"])
                    judged += 1
                    continue

                gt_key = GT_FIELD_MAP[fname]
                truth = gt.get(gt_key)
                score, reason = _score_objective(fname, aug_parsed, truth)
                if score < 0:
                    continue  # not judged
                writer.writerow([args.phase, card_id, repo_id, fname, str(aug_raw)[:80],
                                 int(score * 2), reason,
                                 f"{field.get('attribute_confidence', 0):.3f}"])
                judged += 1
                score_sum += score

            # Freetext fields — LLM judge
            freetext = {fname: runtime_fields[fname] for fname in RUNTIME_FREETEXT_FIELDS
                        if fname in runtime_fields and runtime_fields[fname].get("augmented_value")}
            if not freetext:
                continue

            if gt is None or gt.get("cold_start"):
                for fname, field in freetext.items():
                    writer.writerow([args.phase, card_id, repo_id, fname,
                                     str(field.get("augmented_value"))[:80], 0,
                                     "hallucinated under cold-start",
                                     f"{field.get('attribute_confidence', 0):.3f}"])
                    judged += 1
                continue

            if token is None:
                token = fetch_tapis_token()
                print("Token acquired.\n")

            # Pull low-score image summary for the judge prompt
            try:
                import patra_query
                low_score_imgs = patra_query.get_experiment_images(
                    gt["reference_experiments"], only_low_score=True, limit=6)
                low_score_str = json.dumps(
                    [{"image_name": i["image_name"], "top_label": i["top_label"],
                      "top_probability": i["top_probability"],
                      "ground_truth": i["ground_truth"]} for i in low_score_imgs],
                    indent=2) or "(none)"
            except Exception as e:
                low_score_str = f"(unavailable: {e})"

            category = ""
            input_type = ""
            name = repo_id.split("/")[-1]
            # Walk static fields to surface category/input_type
            for f in r.get("fields", []):
                if f["field_name"] == "category" and f.get("augmented_value"):
                    category = f["augmented_value"]
                elif f["field_name"] == "input_type" and f.get("augmented_value"):
                    input_type = f["augmented_value"]

            print(f"  [{card_id}] {repo_id} — freetext judge ({len(freetext)} fields)")
            try:
                judge_results = _call_runtime_freetext_judge(
                    token, json.dumps(summary_for_judge, indent=2), low_score_str,
                    name, category, input_type,
                    {fname: field["augmented_value"] for fname, field in freetext.items()})
                api_calls += 1
            except Exception as e:
                print(f"    Judge error: {e}")
                for fname, field in freetext.items():
                    writer.writerow([args.phase, card_id, repo_id, fname,
                                     str(field.get("augmented_value"))[:80], -1,
                                     f"Judge error: {e}",
                                     f"{field.get('attribute_confidence', 0):.3f}"])
                continue

            for fname, field in freetext.items():
                result = judge_results.get(fname, {})
                score = result.get("score", -1)
                reason = result.get("reason", "not judged")
                writer.writerow([args.phase, card_id, repo_id, fname,
                                 str(field.get("augmented_value"))[:80], score, reason,
                                 f"{field.get('attribute_confidence', 0):.3f}"])
                judged += 1
                if score >= 0:
                    score_sum += score / 2.0

            time.sleep(0.2)

    avg = score_sum / judged if judged else 0.0
    print()
    print(f"=== Runtime Judge Summary ===")
    print(f"  Fields judged:      {judged}")
    print(f"  Freetext API calls: {api_calls}")
    print(f"  Avg Accuracy:       {avg:.3f}")
    print(f"  Scores saved:       {JUDGE_CSV}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, help="Phase label (must match augmentation run)")
    parser.add_argument("--dataset", required=True, help="Dataset JSON file")
    parser.add_argument("--results", default=None, help="Results JSON file (default: results_{phase_slug}.json)")
    parser.add_argument("--runtime", action="store_true",
                        help="Judge runtime_* fields only, using ground-truth exact/tolerance match for objective "
                             "fields and LLM judge for freetext. Requires poc/mock_experiments_ground_truth.json.")
    args = parser.parse_args()

    if args.runtime:
        run_runtime_judge(args)
        return

    dataset_path = Path(__file__).parent / "data" / "inputs" / args.dataset
    if args.results:
        results_path = RESULTS_DIR / args.results
    else:
        phase_slug = args.phase.replace(" ", "_").replace(":", "").lower()
        results_path = RESULTS_DIR / f"results_{phase_slug}.json"

    if not dataset_path.exists():
        print(f"ERROR: {dataset_path} not found")
        sys.exit(1)
    if not results_path.exists():
        print(f"ERROR: {results_path} not found. Run augmentation first.")
        sys.exit(1)

    dataset = json.loads(dataset_path.read_text())
    results = json.loads(results_path.read_text())
    dataset_by_id = {e["id"]: e for e in dataset}

    print(f"=== LLM Judge: {JUDGE_MODEL} (batch mode) ===")
    print(f"Phase: {args.phase}")
    print(f"Results: {results_path.name}")
    print(f"Records: {len(results)}\n")

    token = fetch_tapis_token()
    print("Token acquired.\n")

    write_header = not JUDGE_CSV.exists()
    judged_count = 0
    skipped_count = 0
    score_sum = 0.0
    api_calls = 0

    with open(JUDGE_CSV, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(["phase", "card_id", "repo_id", "field", "value_excerpt", "judge_score", "judge_reason", "attr_confidence"])

        for r in results:
            card_id = r["card_id"]
            repo_id = r["repo_id"]
            entry = dataset_by_id.get(card_id)
            if not entry:
                continue

            hf_json_str = json.dumps(entry["hf_api_response"], indent=2)[:1500]
            readme = entry.get("readme_body") or ""

            # Collect all filled fields for this card
            fields_to_judge = {}
            field_conf = {}
            for f in r.get("fields", []):
                val = f.get("augmented_value") or ""
                if val.strip():
                    fields_to_judge[f["field_name"]] = val[:200]
                    field_conf[f["field_name"]] = f.get("attribute_confidence", 0.0)
                else:
                    skipped_count += 1

            if not fields_to_judge:
                continue

            print(f"  [{card_id}] {repo_id} — {len(fields_to_judge)} fields in 1 call")

            try:
                judge_results = call_judge_batch(token, hf_json_str, readme, fields_to_judge)
                api_calls += 1
            except Exception as e:
                print(f"    Judge error: {e}")
                for fname in fields_to_judge:
                    writer.writerow([args.phase, card_id, repo_id, fname, fields_to_judge[fname][:80], -1, f"Judge error: {e}", f"{field_conf.get(fname, 0):.3f}"])
                continue

            for fname, val in fields_to_judge.items():
                result = judge_results.get(fname, {})
                score = result.get("score", -1)
                reason = result.get("reason", "not judged")
                writer.writerow([args.phase, card_id, repo_id, fname, val[:80], score, reason, f"{field_conf.get(fname, 0):.3f}"])
                judged_count += 1
                if score >= 0:
                    score_sum += score / 2.0

            time.sleep(0.2)

    avg_accuracy = score_sum / judged_count if judged_count else 0.0

    print(f"\n=== Judge Summary ===")
    print(f"  API calls:      {api_calls} (1 per card, not per field)")
    print(f"  Fields judged:  {judged_count}")
    print(f"  Fields skipped: {skipped_count} (empty)")
    print(f"  Avg Accuracy:   {avg_accuracy:.3f}")
    print(f"  Scores saved:   {JUDGE_CSV}")


if __name__ == "__main__":
    main()
