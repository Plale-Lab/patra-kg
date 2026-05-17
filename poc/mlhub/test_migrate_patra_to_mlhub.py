#!/usr/bin/env python3
"""Hermetic tests for migrate_patra_to_mlhub.

No I/O, no network. Validates the deterministic mapping function and the
filename-resolution helper against synthetic Patra ModelCardDetail dicts.

Run:
    pytest poc/test_migrate_patra_to_mlhub.py -v
    # or:
    python poc/test_migrate_patra_to_mlhub.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from migrate_patra_to_mlhub import build_filename, patra_to_mlhub


SAMPLE_CARD = {
    "external_id": 42,
    "uuid": "11111111-2222-3333-4444-555555555555",
    "name": "resnet50-cv",
    "version": "1.0.0",
    "short_description": "ResNet-50 image classifier",
    "full_description": "A ResNet-50 trained on ImageNet, fine-tuned for cats.",
    "keywords": "resnet,vision,classifier",
    "author": "Jane Doe",
    "input_data": "RGB images 224x224",
    "output_data": "1000-class softmax",
    "input_type": "Image",
    "categories": "ImageClassification",
    "citation": "Doe et al. 2024",
    "foundational_model": "ResNet-50",
    "is_private": False,
    "is_gated": False,
    "ai_model": {
        "model_id": 99,
        "name": "resnet50-cv-model",
        "version": "1.0.0-pt",
        "description": "PyTorch checkpoint",
        "owner": "Jane Doe",
        "location": "s3://patra/resnet50.pt",
        "license": "apache-2.0",
        "framework": "pytorch",
        "model_type": "cnn",
        "test_accuracy": 0.91,
    },
}


def test_direct_field_maps():
    out = patra_to_mlhub(SAMPLE_CARD)
    assert out["name"] == "resnet50-cv"
    assert out["version"] == "1.0.0"
    assert out["framework"] == "pytorch"
    assert out["model_type"] == "cnn"
    assert out["license"] == "apache-2.0"
    assert out["task_types"] == ["ImageClassification"]


def test_unmapped_fields_are_null():
    """Patra detail endpoint doesn't expose model_requirements or bias_analysis."""
    out = patra_to_mlhub(SAMPLE_CARD)
    assert out["inference_software_dependencies"] is None
    assert out["bias_evaluation_score"] is None


def test_label_map_patra_legacy_round_trip():
    out = patra_to_mlhub(SAMPLE_CARD)
    legacy = out["label_map"]["patra_legacy"]
    assert legacy["patra_id"] == 42
    assert legacy["patra_uuid"] == "11111111-2222-3333-4444-555555555555"
    assert legacy["author"] == "Jane Doe"
    assert legacy["short_description"] == "ResNet-50 image classifier"
    assert legacy["full_description"].startswith("A ResNet-50")
    assert legacy["citation"] == "Doe et al. 2024"
    assert legacy["foundational_model"] == "ResNet-50"
    assert legacy["is_gated"] is False
    assert legacy["is_private"] is False
    assert legacy["ai_model_full"]["test_accuracy"] == 0.91


def test_version_falls_back_to_ai_model():
    card = {**SAMPLE_CARD, "version": None}
    out = patra_to_mlhub(card)
    assert out["version"] == "1.0.0-pt"


def test_version_top_level_takes_precedence():
    out = patra_to_mlhub(SAMPLE_CARD)
    assert out["version"] == "1.0.0"  # not "1.0.0-pt"


def test_task_types_null_when_no_categories():
    card = {**SAMPLE_CARD, "categories": None}
    out = patra_to_mlhub(card)
    assert out["task_types"] is None


def test_no_ai_model_section():
    card = {**SAMPLE_CARD, "ai_model": None}
    out = patra_to_mlhub(card)
    assert out["framework"] is None
    assert out["model_type"] is None
    assert out["license"] is None
    assert out["label_map"]["patra_legacy"]["ai_model_full"] is None


def test_filename_uses_uuid_when_present():
    assert build_filename(SAMPLE_CARD) == "11111111-2222-3333-4444-555555555555.json"


def test_filename_falls_back_to_mc_id():
    card = {**SAMPLE_CARD, "uuid": None}
    assert build_filename(card) == "mc_42.json"


def test_filename_rejects_malformed_uuid():
    card = {**SAMPLE_CARD, "uuid": "not-a-uuid"}
    assert build_filename(card) == "mc_42.json"


# ---------------------------------------------------------------------------
# Standalone runner so the file works with `python poc/test_migrate_patra_to_mlhub.py`
# (matches poc/test_runtime_formulas.py convention).
# ---------------------------------------------------------------------------

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
