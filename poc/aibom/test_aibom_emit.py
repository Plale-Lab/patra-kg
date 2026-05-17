#!/usr/bin/env python3
"""Hermetic tests for the AIBOM emit POC.

Run:
    .venv/bin/python poc/aibom/test_aibom_emit.py
    # or:
    pytest poc/aibom/test_aibom_emit.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from aibom_emit import patra_to_aibom


RICH_CARD = {
    "external_id": 42,
    "uuid": "11111111-2222-3333-4444-555555555555",
    "name": "MegaDetector",
    "version": "5.0",
    "author": "0000-0001-2345-6789",
    "input_data": "https://lila.science/datasets/camera-traps",
    "input_type": "Image",
    "output_data": "bbox+class",
    "categories": "ObjectDetection",
    "foundational_model": "YOLOv5",
    "is_private": False,
    "is_gated": False,
    "model_artifact_hash": "sha256:abc123",
    "model_requirements": ["torch==2.1.0", "torchvision==0.16.0"],
    "bias_analysis": {"demographic_parity_diff": 0.01},
    "xai_analysis": {"gradcam_coverage": 0.88},
    "limitations": "Diurnal images only.",
    "compliance_tags": ["IRB-approved"],
    "ai_model": {
        "name": "MegaDetector",
        "version": "5.0-pt",
        "owner": "Microsoft AI for Earth",
        "location": "https://github.com/microsoft/CameraTraps/releases/tag/v5.0",
        "license": "MIT",
        "framework": "pytorch",
        "model_type": "cnn",
        "test_accuracy": 0.943,
    },
    "experiments": [
        {
            "edge_device_id": "device-1",
            "f1_score": 0.94,
            "precision": 0.95,
            "recall": 0.93,
            "map_50": 0.94,
            "power_summary": {
                "total_cpu_power_consumption": 8.0,
                "total_gpu_power_consumption": 14.5,
            },
        },
        {
            "edge_device_id": "device-2",
            "f1_score": 0.91,
            "precision": 0.92,
            "recall": 0.90,
            "map_50": 0.92,
            "power_summary": {
                "total_cpu_power_consumption": 12.0,
                "total_gpu_power_consumption": 18.0,
            },
        },
    ],
}


SPARSE_CARD = {
    "external_id": 99,
    "uuid": None,
    "name": "untitled-model",
    "ai_model": None,
    "experiments": [],
}


# --- envelope -----------------------------------------------------------

def test_envelope_format_and_version():
    aibom = patra_to_aibom(RICH_CARD)
    assert aibom["aibom_format"] == "patra-native"
    assert aibom["aibom_version"] == "0.1.0"
    assert aibom["generated_at"]


# --- model section ------------------------------------------------------

def test_model_section_rich():
    m = patra_to_aibom(RICH_CARD)["model"]
    assert m["uuid"] == "11111111-2222-3333-4444-555555555555"
    assert m["patra_id"] == 42
    assert m["name"] == "MegaDetector"
    assert m["version"] == "5.0"  # top-level wins over ai_model.version
    assert m["framework"] == "pytorch"
    assert m["model_type"] == "cnn"
    assert m["foundational_model"] == "YOLOv5"
    assert m["artifact_hash"] == "sha256:abc123"


def test_version_falls_back_to_ai_model():
    card = {**RICH_CARD, "version": None}
    assert patra_to_aibom(card)["model"]["version"] == "5.0-pt"


# --- aggregates ---------------------------------------------------------

def test_energy_aggregates():
    e = patra_to_aibom(RICH_CARD)["energy"]
    assert e["n_runs"] == 2
    assert e["avg_total_cpu_w"] == 10.0   # (8 + 12) / 2
    assert e["avg_total_gpu_w"] == 16.25  # (14.5 + 18) / 2


def test_evaluation_aggregates():
    e = patra_to_aibom(RICH_CARD)["evaluations"]
    assert e["test_accuracy"] == 0.943
    assert e["avg_f1_score"] == 0.925   # (0.94 + 0.91) / 2
    assert e["avg_precision"] == 0.935  # (0.95 + 0.92) / 2
    assert e["avg_recall"] == 0.915     # (0.93 + 0.90) / 2


def test_infrastructure_collects_unique_devices():
    aibom = patra_to_aibom(RICH_CARD)
    assert aibom["infrastructure"]["experiments_count"] == 2
    assert aibom["infrastructure"]["edge_device_ids"] == ["device-1", "device-2"]


# --- authorship / ORCID ------------------------------------------------

def test_orcid_extracted_when_author_is_orcid():
    aibom = patra_to_aibom(RICH_CARD)
    assert aibom["authorship"]["orcid_id"] == "0000-0001-2345-6789"


def test_orcid_none_when_author_is_plain_name():
    card = {**RICH_CARD, "author": "Jane Doe"}
    aibom = patra_to_aibom(card)
    assert aibom["authorship"]["orcid_id"] is None
    assert aibom["authorship"]["author"] == "Jane Doe"


# --- pass-through fields -----------------------------------------------

def test_bias_xai_round_trip():
    aibom = patra_to_aibom(RICH_CARD)
    assert aibom["bias_fairness"] == {"demographic_parity_diff": 0.01}
    assert aibom["explainability"] == {"gradcam_coverage": 0.88}


def test_dependencies_passes_through():
    deps = patra_to_aibom(RICH_CARD)["dependencies"]
    assert deps["model_requirements"] == ["torch==2.1.0", "torchvision==0.16.0"]


def test_compliance_and_limitations_passes_through():
    aibom = patra_to_aibom(RICH_CARD)
    assert aibom["compliance_tags"] == ["IRB-approved"]
    assert aibom["limitations"].startswith("Diurnal")


# --- gap detection -----------------------------------------------------

def test_rich_card_has_no_gaps():
    assert patra_to_aibom(RICH_CARD)["_gaps"] == []


def test_sparse_card_surfaces_all_gaps():
    gaps = set(patra_to_aibom(SPARSE_CARD)["_gaps"])
    expected = {
        "model.artifact_hash",
        "dependencies.model_requirements",
        "infrastructure.experiments",
        "energy",
        "evaluations",
        "license.identifier",
        "bias_fairness",
        "explainability",
        "limitations",
        "compliance_tags",
        "authorship.orcid_id",
    }
    assert expected.issubset(gaps), f"Missing gaps: {expected - gaps}"


def test_partial_card_surfaces_only_missing_gaps():
    """A card with only some AIBOM fields filled should report only the missing ones."""
    card = {
        **RICH_CARD,
        "limitations": None,
        "compliance_tags": None,
        "model_artifact_hash": None,
    }
    gaps = set(patra_to_aibom(card)["_gaps"])
    assert gaps == {"model.artifact_hash", "limitations", "compliance_tags"}


# --- standalone runner -------------------------------------------------

if __name__ == "__main__":
    fns = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"[ok]  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"[err] {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"[err] {fn.__name__}: {type(e).__name__}: {e}")
    sys.exit(0 if failed == 0 else 1)
