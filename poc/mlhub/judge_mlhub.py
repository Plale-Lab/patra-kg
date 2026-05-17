#!/usr/bin/env python3
"""M4 Judge: score Patra → MLHub augmented fields with Qwen3-32B.

Mirrors judge_augmented.py but grounds judgments on the Patra card +
experiments[] + README (rather than HuggingFace API + README). Appends rows
to the shared judge_scores.csv so visualize_metrics.py picks them up
automatically alongside M1/M2/M3 scores.

Usage:
    python poc/judge_mlhub.py --card-id 1
    python poc/judge_mlhub.py
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

POC_DIR = Path(__file__).parent
RESULTS_PATH = POC_DIR / "results_m4_patra_to_mlhub.json"
M3_RESULTS_PATH = POC_DIR / "results_m3_chain-of-thought.json"
EXPERIMENTS_PATH = POC_DIR / "mock_experiments.json"
HF_CARDS_PATH = POC_DIR / "real_hf_cards.json"
JUDGE_CSV = POC_DIR / "judge_scores.csv"
PHASE_LABEL = "M4: Patra→MLHub"

# One-line field descriptions (shown to the judge alongside each value)
MLHUB_FIELD_DESCRIPTIONS = {
    "name": "Model display name",
    "author": "Creator / organization",
    "version": "Model version string",
    "model_type": "Architecture family (cnn, dnn, llm, ...)",
    "license": "License identifier (apache-2.0, mit, ...)",
    "task_types": "MLHub Task enum value(s) describing model capability",
    "inference_software_dependencies": "Array of package requirements for inference",
    "pretraining_datasets": "Dataset(s) used for pretraining",
    "keywords": "Array of search keywords",
    "libraries": "Array of ML frameworks (pytorch, tensorflow, ...)",
    "inference_hardware": "HardwareRequirements{accelerators, memory_gb} for inference",
    "inference_max_energy_consumption_watts": "Peak power draw during inference (watts)",
    "inference_max_latency_ms": "Max acceptable inference latency (milliseconds)",
    "inference_min_throughput": "Min throughput (images/sec)",
    "inference_max_memory_usage_mb": "Peak memory usage during inference (MB)",
    "inference_distributed": "Whether inference runs across multiple devices",
    "image": "URL of a representative model image / thumbnail",
    "multi_modal": "Accepts multiple input modalities",
    "model_inputs": "Array of input tensor specs {data_type, shape}",
    "model_outputs": "Array of output tensor specs",
    "inference_precision": "Numeric precision at inference (fp32, fp16, int8, ...)",
    "inference_max_compute_utilization_percentage": "Peak compute utilization (0-100)",
    "pretrained": "Released as a pretrained checkpoint",
    "edge_optimized": "Designed for edge deployment",
    "quantization_aware": "Trained with quantization-aware training",
    "supports_quantization": "Can be post-training quantized",
    "pruned": "Weights have been pruned",
    "slimmed": "Distilled / size-reduced",
    "training_time": "Training time in hours",
    "training_precision": "Precision during training",
    "training_hardware": "HardwareRequirements for training",
    "training_max_energy_consumption_watts": "Peak training power (watts)",
    "training_distributed": "Trained across multiple devices",
    "finetuning_datasets": "Array of fine-tuning datasets",
    "regulatory": "Array of compliance tags (HIPAA, GDPR, ...)",
    "bias_evaluation_score": "Integer 0-100; higher = less bias",
    "annotations": "Free-form provenance object",
}

BATCH_JUDGE_PROMPT = """You are scoring MLHub `ModelMetadata` fields that were derived from a Patra model
card, its experiments[] deployment history, and the HuggingFace README. Score EACH field below.

## Patra model card (authoritative source of truth)
{patra_card_json}

## Recent experiments sample (source for inference_* runtime fields)
{exps_summary}

## README excerpt (supplementary source)
{readme_excerpt}

## Fields to evaluate
{fields_json}

## Scoring rubric (apply to EACH field)
0 = wrong, misleading, or contradicted by the Patra card / experiments / README
1 = acceptable but imprecise, or inferred from general knowledge only
2 = correct and directly supported by the Patra card, experiments, or README

Return ONLY valid JSON (no prose, no markdown, no thinking tags):
{{
  "field_name_1": {{"score": N, "reason": "one sentence"}},
  "field_name_2": {{"score": N, "reason": "one sentence"}},
  ...
}}"""


def fetch_tapis_token() -> str:
    username = os.getenv("TAPIS_USERNAME", "").strip()
    password = os.getenv("TAPIS_PASSWORD", "").strip()
    if not username or not password:
        print("ERROR: Set TAPIS_USERNAME and TAPIS_PASSWORD in poc/.env", file=sys.stderr)
        sys.exit(1)
    resp = httpx.post(TAPIS_AUTH_URL,
        json={"username": username, "password": password, "grant_type": "password"}, timeout=15)
    resp.raise_for_status()
    return resp.json()["result"]["access_token"]["access_token"]


def call_judge_batch(token: str, patra_card_json: str, exps_summary: str,
                     readme: str, fields_to_judge: dict) -> dict:
    readme_excerpt = readme[:2000] if readme else "(no README available)"
    fields_json = json.dumps(
        {fname: {"value": val, "description": MLHUB_FIELD_DESCRIPTIONS.get(fname, fname)}
         for fname, val in fields_to_judge.items()},
        indent=2,
        default=str,
    )
    prompt = BATCH_JUDGE_PROMPT.format(
        patra_card_json=patra_card_json[:4000],
        exps_summary=exps_summary,
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
            "max_tokens": 2000,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
        timeout=90,
    )
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"].strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


def patra_card_as_dict(card: dict) -> dict:
    return {f["field_name"]: f["augmented_value"] for f in card.get("fields", [])
            if f.get("augmented_value") not in (None, "")}


def summarize_experiments_for_judge(exps: list[dict], devices_by_id: dict) -> str:
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Judge M4 Patra → MLHub output.")
    parser.add_argument("--card-id", type=int, default=None,
                        help="Judge only this card_id; default judges all in the results file.")
    parser.add_argument("--results", type=Path, default=RESULTS_PATH,
                        help="M4 results JSON.")
    args = parser.parse_args()

    m4_results = json.loads(args.results.read_text())
    m3_by_id = {c["card_id"]: c for c in json.loads(M3_RESULTS_PATH.read_text())}
    hf_by_id = {c["id"]: c for c in json.loads(HF_CARDS_PATH.read_text())}
    fixture = json.loads(EXPERIMENTS_PATH.read_text())
    devices_by_id = {d["id"]: d for d in fixture["edge_devices"]}
    experiments_by_repo: dict[str, list[dict]] = {}
    for exp in fixture["experiments"]:
        experiments_by_repo.setdefault(exp["model_id_ckn"], []).append(exp)

    if args.card_id is not None:
        m4_results = [c for c in m4_results if c["card_id"] == args.card_id]
    if not m4_results:
        print(f"ERROR: no M4 results for card_id={args.card_id}", file=sys.stderr)
        return 1

    print(f"Loaded {len(m4_results)} M4 card(s) to judge.")
    token = fetch_tapis_token()
    print(f"Token acquired.\n")

    write_header = not JUDGE_CSV.exists()
    judged = 0
    score_sum = 0.0
    api_calls = 0

    with open(JUDGE_CSV, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(["phase", "card_id", "repo_id", "field", "value_excerpt",
                             "judge_score", "judge_reason", "attr_confidence"])

        for r in m4_results:
            card_id = r["card_id"]
            repo_id = r["repo_id"]
            m3_card = m3_by_id.get(card_id)
            if not m3_card:
                print(f"  [#{card_id}] no M3 source — skipping", file=sys.stderr)
                continue
            patra_dict = patra_card_as_dict(m3_card)
            patra_card_json = json.dumps(patra_dict, indent=2, default=str)
            readme = hf_by_id.get(card_id, {}).get("readme_body", "")
            exps = experiments_by_repo.get(repo_id, [])
            exps_summary = summarize_experiments_for_judge(exps, devices_by_id)

            # Collect filled fields
            fields_to_judge = {}
            field_conf: dict[str, float] = {}
            for f in r.get("fields", []):
                val = f.get("augmented_value") or ""
                if val.strip():
                    fields_to_judge[f["field_name"]] = val[:200]
                    field_conf[f["field_name"]] = f.get("attribute_confidence", 0.0)
            if not fields_to_judge:
                print(f"  [#{card_id}] no filled fields to judge")
                continue

            print(f"  [#{card_id}] {repo_id} — {len(fields_to_judge)} fields in 1 call")
            try:
                judge_results = call_judge_batch(token, patra_card_json, exps_summary,
                                                  readme, fields_to_judge)
                api_calls += 1
            except Exception as e:
                print(f"    Judge error: {e}", file=sys.stderr)
                for fname in fields_to_judge:
                    writer.writerow([PHASE_LABEL, card_id, repo_id, fname,
                                     fields_to_judge[fname][:80], -1, f"Judge error: {e}",
                                     f"{field_conf.get(fname, 0):.3f}"])
                continue

            for fname, val in fields_to_judge.items():
                result = judge_results.get(fname, {})
                score = result.get("score", -1)
                reason = result.get("reason", "not judged")
                writer.writerow([PHASE_LABEL, card_id, repo_id, fname, val[:80],
                                 score, reason, f"{field_conf.get(fname, 0):.3f}"])
                judged += 1
                if score >= 0:
                    score_sum += score / 2.0

            time.sleep(0.2)

    avg_accuracy = score_sum / judged if judged else 0.0
    print(f"\n=== Judge Summary ===")
    print(f"  API calls:      {api_calls}")
    print(f"  Fields judged:  {judged}")
    print(f"  Avg accuracy:   {avg_accuracy:.3f}")
    print(f"  Scores saved:   {JUDGE_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
