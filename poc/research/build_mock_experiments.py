#!/usr/bin/env python3
"""Build poc/mock_experiments.json + poc/mock_experiments_ground_truth.json.

Generates a deterministic fixture that models Patra's normalized shape
(post Neo4j/Postgres Kafka Connector): experiments[], experiment_images[],
raw_images[], edge_devices[], plus hf_repo_id <-> model_id mapping.

Ground truth is derived by the same formulas M4 uses at augment time, so the
oracle is guaranteed consistent with the data.

Usage:
    python poc/build_mock_experiments.py
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import runtime_formulas

HERE = Path(__file__).parent
HF_CARDS_PATH = HERE / "data" / "inputs" / "real_hf_cards.json"
FIXTURE_PATH = HERE / "data" / "inputs" / "mock_experiments.json"
GROUND_TRUTH_PATH = HERE / "data" / "inputs" / "mock_experiments_ground_truth.json"

# ---------------------------------------------------------------------------
# Edge devices (matches edge_devices table in Patra schema)
# ---------------------------------------------------------------------------

EDGE_DEVICES = [
    {"device_id": "jetson-nano-1", "device_type": "Jetson Nano", "ram_mb": 4096,  "class": "edge",       "gpu": True},
    {"device_id": "a100-1",        "device_type": "A100",        "ram_mb": 81920, "class": "datacenter", "gpu": True},
    {"device_id": "cpu-x86-1",     "device_type": "CPU-x86",     "ram_mb": 16384, "class": "cpu",        "gpu": False},
]
DEVICE_BY_ID = {d["device_id"]: d for d in EDGE_DEVICES}

# ---------------------------------------------------------------------------
# Card-level metadata: category + input_type + experiment recipe
# Drives which cards are "similar" under (category, input_type) and how many
# experiments each generates, on which devices.
# ---------------------------------------------------------------------------

CARD_RECIPES: dict[int, dict] = {
    # NLP + Text cluster (cards 1, 3, 5, 6, 8) — 5 cards share this bucket.
    # #1 sentence-similarity -> NLP/Text via the generate_synthetic_dataset pipeline map.
    1: {"category": "natural language processing", "input_type": "Text",
        "devices": ["cpu-x86-1", "cpu-x86-1", "a100-1"], "f1_base": 0.82, "latency_ms": [18, 22, 26], "users": ["neelk"]},
    # #2 image-text-to-text -> NLP/Multimodal. Singleton bucket -> cold-start.
    2: {"category": "natural language processing", "input_type": "Multimodal",
        "devices": ["a100-1", "a100-1"], "f1_base": 0.70, "latency_ms": [140, 160], "users": ["neelk"]},
    3: {"category": "natural language processing", "input_type": "Text",
        "devices": ["a100-1", "cpu-x86-1", "cpu-x86-1", "a100-1"], "f1_base": 0.88, "latency_ms": [45, 90, 88, 42], "users": ["swithana", "neelk"]},
    # #4 electra-base-discriminator has no pipeline_tag -> cold-start (no recipe).
    5: {"category": "natural language processing", "input_type": "Text",
        "devices": ["cpu-x86-1", "a100-1", "cpu-x86-1"], "f1_base": 0.80, "latency_ms": [24, 20, 28], "users": ["neelk"]},
    6: {"category": "natural language processing", "input_type": "Text",
        "devices": ["a100-1", "a100-1"], "f1_base": 0.75, "latency_ms": [210, 245], "users": ["beckstei"]},
    # #7 image-classification -> "classification"/Image (not "computer vision"). Singleton -> cold-start.
    7: {"category": "classification", "input_type": "Image",
        "devices": ["jetson-nano-1", "jetson-nano-1", "cpu-x86-1", "jetson-nano-1"], "f1_base": 0.92, "latency_ms": [35, 40, 60, 38], "users": ["neelk", "beckstei"]},
    8: {"category": "natural language processing", "input_type": "Text",
        "devices": ["a100-1", "cpu-x86-1"], "f1_base": 0.45, "latency_ms": [55, 95], "users": ["swithana"]},  # low f1 -> failure-mode grounding
    # #9 automatic-speech-recognition -> "classification"/Audio. Singleton -> cold-start.
    9: {"category": "classification", "input_type": "Audio",
        "devices": ["a100-1", "a100-1", "a100-1"], "f1_base": 0.72, "latency_ms": [180, 175, 185], "users": ["beckstei"]},
    # #10 object-detection -> "computer vision"/Image. Singleton -> cold-start.
    10: {"category": "computer vision", "input_type": "Image",
         "devices": ["jetson-nano-1", "jetson-nano-1", "a100-1", "jetson-nano-1"], "f1_base": 0.88, "latency_ms": [22, 25, 12, 28], "users": ["neelk", "beckstei"]},
}

USERS = [
    {"username": "neelk"},
    {"username": "swithana"},
    {"username": "beckstei"},
]

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def build_experiments(rng: random.Random, cards: list[dict]) -> tuple[list[dict], dict[int, int]]:
    """Generate experiment rows + hf_repo_id -> model_id mapping."""
    model_id_map: dict[int, int] = {}  # hf_card_id -> synthetic model_id
    experiments: list[dict] = []
    next_model_id = 100
    next_exp_id = 1
    base_ts = datetime(2026, 3, 1, 8, 0, 0, tzinfo=timezone.utc)

    for card in cards:
        card_id = card["id"]
        if card["asset_type"] != "model_card":
            continue
        recipe = CARD_RECIPES.get(card_id)
        if recipe is None:
            continue
        model_id = next_model_id
        model_id_map[card_id] = model_id
        next_model_id += 1

        for run_idx, (device_id, lat_ms) in enumerate(zip(recipe["devices"], recipe["latency_ms"])):
            # F1 with realistic per-run noise (±0.06 around the base)
            f1 = max(0.02, min(0.99, recipe["f1_base"] + rng.uniform(-0.06, 0.06)))
            # Precision/recall derived from f1 with small asymmetry
            precision = max(0.02, min(0.99, f1 + rng.uniform(-0.04, 0.04)))
            recall = max(0.02, min(0.99, f1 + rng.uniform(-0.04, 0.04)))
            total_images = rng.choice([100, 200, 500, 1000])
            total_gt = total_images
            true_positives = int(recall * total_gt)
            false_negatives = total_gt - true_positives
            false_positives = max(0, int(true_positives * (1 - precision) / max(precision, 0.01)))
            total_predictions = true_positives + false_positives

            # Power draw: scales with device class
            device = DEVICE_BY_ID[device_id]
            if device["class"] == "datacenter":
                cpu_w = rng.uniform(6.0, 9.0)
                gpu_w = rng.uniform(180.0, 320.0)  # A100 under load
            elif device["class"] == "edge":
                cpu_w = rng.uniform(2.0, 3.5)
                gpu_w = rng.uniform(4.0, 9.0)  # Jetson integrated GPU
            else:
                cpu_w = rng.uniform(14.0, 22.0)
                gpu_w = 0.0

            start_at = base_ts + timedelta(days=len(experiments), hours=rng.randint(0, 18))
            submitted_at = start_at + timedelta(seconds=rng.randint(1, 20))
            executed_at = submitted_at + timedelta(milliseconds=lat_ms * total_images)

            user_id = recipe["users"][run_idx % len(recipe["users"])]

            ckn_exp_id = f"{card['repo_id'].split('/')[-1].lower()}-run{run_idx + 1}"[:80]

            experiments.append({
                "id": next_exp_id,
                "experiment_uid": ckn_exp_id,
                "user_id": user_id,
                "edge_device_id": device_id,
                "device_type_ckn": device["device_type"].lower().replace(" ", "-"),
                "model_id": model_id,
                "model_id_ckn": card["repo_id"],

                "start_at": start_at.isoformat(),
                "executed_at": executed_at.isoformat(),

                "total_images": total_images,
                "total_predictions": total_predictions,
                "total_ground_truth_objects": total_gt,
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "precision": round(precision, 5),
                "recall": round(recall, 5),
                "f1_score": round(f1, 5),
                "mean_iou": None,
                "map_50": None,
                "map_50_95": None,

                "total_cpu_power_w": round(cpu_w, 3),
                "total_gpu_power_w": round(gpu_w, 3),
                "per_image_latency_ms": lat_ms,  # useful for exact-match oracle
            })
            next_exp_id += 1

    return experiments, model_id_map


def build_experiment_images(rng: random.Random, experiments: list[dict]) -> tuple[list[dict], list[dict]]:
    """Build ~40 experiment_images (attached to low-performing experiments) and ~10 raw_images."""
    # Low-performing experiments = f1 < 0.6
    low_perf = [e for e in experiments if e["f1_score"] < 0.6]
    # If not enough, also take the bottom quartile
    if len(low_perf) < 3:
        sorted_by_f1 = sorted(experiments, key=lambda e: e["f1_score"])
        low_perf = sorted_by_f1[: max(3, len(sorted_by_f1) // 4)]

    raw_images: list[dict] = []
    for i in range(10):
        raw_images.append({
            "id": 200 + i,
            "image_name": f"sample_{i:03d}.jpg",
            "ground_truth": {"label": rng.choice(["code_python", "code_java", "night_scene", "wildlife_empty",
                                                   "animal_deer", "urban_litter", "speech_en", "speech_km"])},
        })

    exp_images: list[dict] = []
    next_id = 1000
    for exp in low_perf:
        # 8-12 low-score images per low-perf experiment
        n_imgs = rng.randint(8, 12)
        for _ in range(n_imgs):
            raw = rng.choice(raw_images)
            # Predicted label often wrong for low-perf runs
            labels = ["code_python", "code_java", "night_scene", "wildlife_empty",
                     "animal_deer", "urban_litter", "speech_en", "speech_km"]
            predicted = rng.choice(labels)
            gt_label = raw["ground_truth"]["label"]
            # Lower top_probability when wrong
            if predicted == gt_label:
                top_prob = rng.uniform(0.55, 0.75)
            else:
                top_prob = rng.uniform(0.22, 0.48)

            recv = datetime.fromisoformat(exp["start_at"]) + timedelta(seconds=rng.randint(1, 60))
            scored = recv + timedelta(milliseconds=exp["per_image_latency_ms"])

            exp_images.append({
                "id": next_id,
                "experiment_id": exp["id"],
                "raw_image_id": raw["id"],
                "image_name": raw["image_name"],
                "ground_truth": raw["ground_truth"],
                "image_count": 1,
                "image_received_at": recv.isoformat(),
                "image_scored_at": scored.isoformat(),
                "image_store_deleted_at": (scored + timedelta(seconds=1)).isoformat(),
                "image_decision": "Deleted" if top_prob < 0.5 else "Saved",
                "top_label": predicted,
                "top_probability": round(top_prob, 7),
                "ingested_at": exp["executed_at"],
                "scores": [
                    {"label": predicted, "probability": round(top_prob, 5)},
                    {"label": rng.choice(labels), "probability": round(top_prob * 0.7, 5)},
                    {"label": rng.choice(labels), "probability": round(top_prob * 0.5, 5)},
                ],
            })
            next_id += 1

    return exp_images, raw_images


# ---------------------------------------------------------------------------
# Ground truth derivation — delegates to runtime_formulas (single source of
# truth shared with augment_runtime.py so the oracle cannot drift from the
# augmentation output).
# ---------------------------------------------------------------------------

_VALID_DEVICE_TYPES = {d["device_type"] for d in EDGE_DEVICES}


def derive_ground_truth(cards: list[dict], experiments: list[dict], model_id_map: dict[int, int]) -> dict:
    """Derive per-card expected runtime values from the fixture."""
    truth: dict[str, dict] = {}

    for card in cards:
        card_id = card["id"]
        if card["asset_type"] != "model_card":
            continue
        recipe = CARD_RECIPES.get(card_id)
        if recipe is None:
            continue

        # Similar cards = same (category, input_type), excluding self
        similar_card_ids = [
            c["id"] for c in cards
            if c["asset_type"] == "model_card"
            and c["id"] != card_id
            and CARD_RECIPES.get(c["id"], {}).get("category") == recipe["category"]
            and CARD_RECIPES.get(c["id"], {}).get("input_type") == recipe["input_type"]
        ]
        similar_model_ids = [model_id_map[cid] for cid in similar_card_ids if cid in model_id_map]
        evidence = [e for e in experiments if e["model_id"] in similar_model_ids]

        if not evidence:
            # Cold-start: all runtime fields null
            truth[str(card_id)] = {
                "repo_id": card["repo_id"],
                "similar_model_ids": [],
                "reference_experiments": [],
                "cold_start": True,
                "suggested_hardware": None,
                "expected_f1_range": None,
                "expected_latency_ms": None,
                "deployment_maturity": None,
                "recommended_min_ram_mb": None,
                "inference_cost_class": None,
                "expected_total_power_w": None,
            }
            continue

        truth[str(card_id)] = {
            "repo_id": card["repo_id"],
            "similar_model_ids": similar_model_ids,
            "reference_experiments": [e["id"] for e in evidence],
            "cold_start": False,
            "suggested_hardware": runtime_formulas.suggested_hardware(evidence, DEVICE_BY_ID, _VALID_DEVICE_TYPES),
            "expected_f1_range": runtime_formulas.expected_f1_range(evidence),
            "expected_latency_ms": runtime_formulas.expected_latency_ms(evidence),
            "deployment_maturity": runtime_formulas.deployment_maturity(evidence),
            "recommended_min_ram_mb": runtime_formulas.recommended_min_ram_mb(evidence, DEVICE_BY_ID),
            "inference_cost_class": runtime_formulas.inference_cost_class(evidence, DEVICE_BY_ID),
            "expected_total_power_w": runtime_formulas.expected_total_power_w(evidence),
        }

    return truth


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not HF_CARDS_PATH.exists():
        raise SystemExit(f"ERROR: {HF_CARDS_PATH} not found. Run fetch_real_hf_cards.py first.")
    cards = json.loads(HF_CARDS_PATH.read_text())
    rng = random.Random(42)

    experiments, model_id_map = build_experiments(rng, cards)
    exp_images, raw_images = build_experiment_images(rng, experiments)
    ground_truth = derive_ground_truth(cards, experiments, model_id_map)

    fixture = {
        "edge_devices": EDGE_DEVICES,
        "users": USERS,
        "model_id_map": {str(hf_id): mid for hf_id, mid in model_id_map.items()},
        "card_category_index": {
            str(card_id): {
                "category": recipe["category"],
                "input_type": recipe["input_type"],
                "model_id": model_id_map.get(card_id),
            }
            for card_id, recipe in CARD_RECIPES.items()
        },
        "experiments": experiments,
        "experiment_images": exp_images,
        "raw_images": raw_images,
    }

    FIXTURE_PATH.write_text(json.dumps(fixture, indent=2))
    GROUND_TRUTH_PATH.write_text(json.dumps(ground_truth, indent=2))

    # Summary
    print(f"Wrote {FIXTURE_PATH.name}: {len(experiments)} experiments, "
          f"{len(exp_images)} experiment_images, {len(raw_images)} raw_images")
    print(f"Wrote {GROUND_TRUTH_PATH.name}: ground truth for {len(ground_truth)} cards")
    print()
    cold = sum(1 for v in ground_truth.values() if v["cold_start"])
    print(f"  Cold-start cards (no similar models): {cold}")
    print(f"  Cards with evidence:                  {len(ground_truth) - cold}")
    print()
    print("Per-card ground truth:")
    for cid, gt in ground_truth.items():
        if gt["cold_start"]:
            print(f"  #{cid:>2} [{gt['repo_id'][:40]:<40}]  COLD-START")
        else:
            print(f"  #{cid:>2} [{gt['repo_id'][:40]:<40}]  "
                  f"hw={gt['suggested_hardware']:<12}  "
                  f"f1={gt['expected_f1_range']}  "
                  f"lat={gt['expected_latency_ms']}ms  "
                  f"maturity={gt['deployment_maturity']:<12}  "
                  f"cost={gt['inference_cost_class']:<6}  "
                  f"refs={len(gt['reference_experiments'])}")


if __name__ == "__main__":
    main()
