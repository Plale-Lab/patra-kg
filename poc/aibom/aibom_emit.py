#!/usr/bin/env python3
"""Patra-native AIBOM JSON emitter (POC).

Takes a Patra ModelCardDetail JSON and emits a Patra-native AIBOM JSON.
Deterministic structural mapping; no DB writes; no external lookups.

This POC demonstrates the headline §F.1 recommendation from
DATA_AUGMENTATION_RESEARCH.md: emit AIBOM at zero schema-change cost,
surfacing gaps in `_gaps` so the consumer sees exactly what isn't filled
yet (compliance_tags, limitations, model_artifact_hash, bias_analysis,
xai_analysis are all real schema gaps today).

Usage:
    python poc/aibom/aibom_emit.py --input poc/aibom/sample_card.json
    python poc/aibom/aibom_emit.py --input card.json --output-dir /tmp/aibom

Output: <output-dir>/<uuid-or-mc_id>.aibom.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_OUTPUT_DIR = Path(__file__).parent / "aibom_output"
AIBOM_FORMAT = "patra-native"
AIBOM_VERSION = "0.1.0"


def _safe_get(d: dict, *path, default=None):
    cur = d
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return cur if cur is not None else default


def _avg(values: list) -> float | None:
    real = [v for v in values if v is not None]
    return sum(real) / len(real) if real else None


def _is_orcid(text) -> bool:
    """Loose ORCID iD shape: 0000-0000-0000-0000."""
    if not isinstance(text, str):
        return False
    parts = text.replace(" ", "").split("-")
    return len(parts) == 4 and all(len(p) == 4 for p in parts)


def patra_to_aibom(card: dict) -> dict:
    """Map a Patra card to a Patra-native AIBOM JSON object.

    Every AIBOM concept either has a stable Patra-side source field or
    lands in `_gaps` so consumers see what's missing.
    """
    ai_model = card.get("ai_model") or {}
    experiments = card.get("experiments") or []

    aibom = {
        "aibom_format": AIBOM_FORMAT,
        "aibom_version": AIBOM_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "uuid": card.get("uuid"),
            "patra_id": card.get("external_id"),
            "name": card.get("name"),
            "version": card.get("version") or ai_model.get("version"),
            "location": ai_model.get("location"),
            "framework": ai_model.get("framework"),
            "model_type": ai_model.get("model_type"),
            "foundational_model": card.get("foundational_model"),
            "artifact_hash": card.get("model_artifact_hash"),
        },
        "training_data": {
            "input_data_uri": card.get("input_data"),
            "input_type": card.get("input_type"),
            "output_data_uri": card.get("output_data"),
        },
        "dependencies": {
            "model_requirements": card.get("model_requirements") or [],
        },
        "infrastructure": {
            "experiments_count": len(experiments),
            "edge_device_ids": sorted({
                e.get("edge_device_id") for e in experiments if e.get("edge_device_id")
            }),
        },
        "energy": {
            "avg_total_cpu_w": _avg([
                _safe_get(e, "power_summary", "total_cpu_power_consumption") for e in experiments
            ]),
            "avg_total_gpu_w": _avg([
                _safe_get(e, "power_summary", "total_gpu_power_consumption") for e in experiments
            ]),
            "n_runs": len(experiments),
        },
        "evaluations": {
            "test_accuracy": ai_model.get("test_accuracy"),
            "avg_f1_score": _avg([e.get("f1_score") for e in experiments]),
            "avg_precision": _avg([e.get("precision") for e in experiments]),
            "avg_recall": _avg([e.get("recall") for e in experiments]),
            "avg_map_50": _avg([e.get("map_50") for e in experiments]),
        },
        "license": {
            "identifier": ai_model.get("license"),
            "is_gated": bool(card.get("is_gated")),
            "is_private": bool(card.get("is_private")),
        },
        "bias_fairness": card.get("bias_analysis"),
        "explainability": card.get("xai_analysis"),
        "limitations": card.get("limitations"),
        "compliance_tags": card.get("compliance_tags"),
        "authorship": {
            "author": card.get("author"),
            "orcid_id": card.get("author") if _is_orcid(card.get("author")) else None,
            "owner": ai_model.get("owner"),
        },
    }

    aibom["_gaps"] = _detect_gaps(aibom)
    return aibom


def _detect_gaps(aibom: dict) -> list[str]:
    gaps = []
    if not aibom["model"]["artifact_hash"]:
        gaps.append("model.artifact_hash")
    if not aibom["dependencies"]["model_requirements"]:
        gaps.append("dependencies.model_requirements")
    if aibom["infrastructure"]["experiments_count"] == 0:
        gaps.append("infrastructure.experiments")
    if aibom["energy"]["n_runs"] == 0:
        gaps.append("energy")
    if aibom["evaluations"]["test_accuracy"] is None and aibom["evaluations"]["avg_f1_score"] is None:
        gaps.append("evaluations")
    if not aibom["license"]["identifier"]:
        gaps.append("license.identifier")
    if not aibom["bias_fairness"]:
        gaps.append("bias_fairness")
    if not aibom["explainability"]:
        gaps.append("explainability")
    if not aibom["limitations"]:
        gaps.append("limitations")
    if not aibom["compliance_tags"]:
        gaps.append("compliance_tags")
    if not aibom["authorship"]["orcid_id"]:
        gaps.append("authorship.orcid_id")
    return gaps


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Emit a Patra-native AIBOM JSON from a Patra Model Card."
    )
    parser.add_argument("--input", type=Path, required=True,
                        help="Path to a Patra ModelCardDetail JSON file")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    args = parser.parse_args(argv)

    card = json.loads(args.input.read_text())
    aibom = patra_to_aibom(card)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    name = card.get("uuid") or f"mc_{card.get('external_id', 'unknown')}"
    out_path = args.output_dir / f"{name}.aibom.json"
    out_path.write_text(json.dumps(aibom, indent=2, default=str))

    print(f"AIBOM written: {out_path}")
    print(f"Gaps detected ({len(aibom['_gaps'])}): {aibom['_gaps']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
