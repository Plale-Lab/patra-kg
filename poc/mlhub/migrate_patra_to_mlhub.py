#!/usr/bin/env python3
"""ETL: Patra Model Card -> mock MLHub Model Metadata.

Reads from the live Patra REST API (`GET /modelcards`, `GET /modelcard/{id}`)
and writes one MLHub `ModelMetadata` JSON per card into a local "mock MLHub"
directory. No HTTP traffic to MLHub; no commits; no pushes.

Usage:
  python poc/migrate_patra_to_mlhub.py --id 42
  python poc/migrate_patra_to_mlhub.py --all
  python poc/migrate_patra_to_mlhub.py --all --output-dir /tmp/mlhub_mock
  python poc/migrate_patra_to_mlhub.py --id 42 --augment        # opt into LLM

Env vars:
  PATRA_TOKEN   Bearer token for Patra REST (optional; without it, only public
                cards are visible).
  TAPIS_TOKEN   Required only when --augment is passed; consumed by the LLM
                pipeline in poc/augment_mlhub.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import httpx


DEFAULT_PATRA_BASE = "https://patradb.pods.icicleai.tapis.io"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "mlhub_mock"

# Patra fields that exist on the create payload but are NOT exposed by the
# detail endpoint. Recorded here so the gap is auditable; mapping leaves the
# corresponding MLHub fields null.
PATRA_DETAIL_GAPS = ("model_metrics", "bias_analysis", "xai_analysis", "model_requirements")


def patra_to_mlhub(card: dict) -> dict:
    """Deterministic Patra ModelCardDetail -> MLHub ModelMetadata mapping.

    Direct field maps populate name/version/framework/license/task_types.
    Patra-only fields are stashed under label_map.patra_legacy so nothing is lost.
    Fields MLHub has but Patra doesn't expose stay null (see PATRA_DETAIL_GAPS).
    """
    ai_model = card.get("ai_model") or {}
    categories = card.get("categories")

    return {
        "name": card.get("name"),
        "version": card.get("version") or ai_model.get("version"),
        "framework": ai_model.get("framework"),
        "model_type": ai_model.get("model_type"),
        "license": ai_model.get("license"),
        "task_types": [categories] if categories else None,
        # Below: Patra detail endpoint doesn't return model_requirements / bias_analysis,
        # so these stay null. See PATRA_DETAIL_GAPS.
        "inference_software_dependencies": None,
        "bias_evaluation_score": None,
        "label_map": {
            "patra_legacy": {
                "patra_id": card.get("external_id"),
                "patra_uuid": card.get("uuid"),
                "author": card.get("author"),
                "short_description": card.get("short_description"),
                "full_description": card.get("full_description"),
                "keywords": card.get("keywords"),
                "input_data": card.get("input_data"),
                "output_data": card.get("output_data"),
                "input_type": card.get("input_type"),
                "citation": card.get("citation"),
                "foundational_model": card.get("foundational_model"),
                "is_gated": card.get("is_gated"),
                "is_private": card.get("is_private"),
                "ai_model_full": ai_model or None,
            }
        },
    }


def maybe_augment(card: dict, mlhub: dict, enabled: bool) -> dict:
    """If --augment is set, fold LLM-inferred fields over the deterministic dict.

    Deterministic-mapped fields are NEVER overwritten; LLM only fills nulls.
    """
    if not enabled:
        return mlhub
    sys.path.insert(0, str(Path(__file__).parent))
    import augment_mlhub as augm  # imported lazily so default path has no Tapis dep

    tapis_token = os.environ.get("TAPIS_TOKEN")
    if not tapis_token:
        raise RuntimeError("--augment requires TAPIS_TOKEN env var")

    augmented_fields, _latency_ms = augm.call_llm_mlhub(
        token=tapis_token,
        patra_card_json=json.dumps(card),
        readme_body="",
        exps_summary="",
    )
    for key, value in augmented_fields.items():
        if mlhub.get(key) is None:
            mlhub[key] = value
    return mlhub


def build_filename(card: dict) -> str:
    """Patra UUID if available; bigint id fallback otherwise."""
    uuid = card.get("uuid")
    if isinstance(uuid, str) and len(uuid) == 36:
        return f"{uuid}.json"
    return f"mc_{card['external_id']}.json"


def fetch_card(client: httpx.Client, base_url: str, mc_id: int) -> dict:
    response = client.get(f"{base_url.rstrip('/')}/modelcard/{mc_id}")
    response.raise_for_status()
    return response.json()


def enumerate_card_ids(client: httpx.Client, base_url: str, limit: int) -> Iterable[int]:
    skip = 0
    base = base_url.rstrip("/")
    while True:
        response = client.get(f"{base}/modelcards", params={"skip": skip, "limit": limit})
        response.raise_for_status()
        page = response.json()
        if not page:
            return
        for summary in page:
            yield int(summary["mc_id"])
        if len(page) < limit:
            return
        skip += limit


def write_card_file(metadata: dict, out_dir: Path, filename: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    out_path.write_text(json.dumps(metadata, indent=2, default=str))
    return out_path


def write_report(report: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "migration_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    return out_path


def make_client(token: str | None) -> httpx.Client:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return httpx.Client(headers=headers, timeout=30.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="ETL: Patra Model Card -> mock MLHub Model Metadata.",
    )
    parser.add_argument("--patra-base-url", default=DEFAULT_PATRA_BASE)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--id", type=int, help="Single Patra mc_id to migrate")
    selector.add_argument("--all", action="store_true", help="Enumerate every visible card")
    parser.add_argument("--limit", type=int, default=100, help="Page size for --all")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--augment", action="store_true",
                        help="Run LLM augmentation after deterministic mapping")
    parser.add_argument("--fail-fast", action="store_true",
                        help="Stop on first error instead of continuing")
    args = parser.parse_args(argv)

    token = os.environ.get("PATRA_TOKEN")
    client = make_client(token)

    if args.id is not None:
        ids: Iterable[int] = iter([args.id])
    else:
        ids = enumerate_card_ids(client, args.patra_base_url, args.limit)

    results: list[dict] = []
    attempted = succeeded = failed = 0

    try:
        for mc_id in ids:
            attempted += 1
            try:
                card = fetch_card(client, args.patra_base_url, mc_id)
                mlhub = patra_to_mlhub(card)
                mlhub = maybe_augment(card, mlhub, args.augment)
                filename = build_filename(card)
                path = write_card_file(mlhub, args.output_dir, filename)
                results.append({
                    "mc_id": mc_id,
                    "uuid": card.get("uuid"),
                    "filename": filename,
                    "status": "ok",
                })
                succeeded += 1
                print(f"[ok]  mc_id={mc_id} -> {path}")
            except Exception as exc:  # noqa: BLE001 — top-level per-card guard
                failed += 1
                results.append({
                    "mc_id": mc_id,
                    "uuid": None,
                    "filename": None,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                })
                print(f"[err] mc_id={mc_id}: {exc}", file=sys.stderr)
                if args.fail_fast:
                    break
    finally:
        client.close()

    report = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "patra_base_url": args.patra_base_url,
        "augment_enabled": args.augment,
        "attempted": attempted,
        "succeeded": succeeded,
        "failed": failed,
        "results": results,
    }
    report_path = write_report(report, args.output_dir)
    print(f"\nReport: {report_path}")
    print(f"Attempted={attempted} Succeeded={succeeded} Failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
