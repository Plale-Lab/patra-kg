"""MCP-compatible query layer for the runtime augmentation POC.

Signatures mirror the tools in `mcp_server/main.py` (search_modelcards,
list_user_experiments, get_experiment_detail, get_experiment_power,
get_experiment_images) so the production path can swap the import for a real
MCP client with no caller changes.

Reads poc/mock_experiments.json. Run poc/build_mock_experiments.py first.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

FIXTURE_PATH = Path(__file__).parent / "data" / "inputs" / "mock_experiments.json"


@lru_cache(maxsize=1)
def _load() -> dict:
    if not FIXTURE_PATH.exists():
        raise FileNotFoundError(
            f"{FIXTURE_PATH.name} not found. Run poc/build_mock_experiments.py first."
        )
    return json.loads(FIXTURE_PATH.read_text())


def _index() -> dict:
    return _load()["card_category_index"]


def _experiments() -> list[dict]:
    return _load()["experiments"]


def _experiment_images() -> list[dict]:
    return _load()["experiment_images"]


def _devices_by_id() -> dict[int, dict]:
    return {d["id"]: d for d in _load()["edge_devices"]}


# ---------------------------------------------------------------------------
# Public API — mirrors mcp_server/main.py tool signatures
# ---------------------------------------------------------------------------

def search_similar_models(category: str, input_type: str, limit: int = 5,
                          exclude_hf_card_id: int | None = None) -> list[dict]:
    """Return model_cards with matching (category, input_type).

    Similarity rule: exact match on both fields. Self-exclusion if
    exclude_hf_card_id is given (matches production behavior when augmenting
    a new card that shouldn't reference itself).
    """
    index = _index()
    matches = []
    for card_id_str, info in index.items():
        card_id = int(card_id_str)
        if card_id == exclude_hf_card_id:
            continue
        if info["category"] == category and info["input_type"] == input_type:
            matches.append({
                "hf_card_id": card_id,
                "model_id": info["model_id"],
                "category": info["category"],
                "input_type": info["input_type"],
            })
    return matches[:limit]


def get_experiments_for_models(model_ids: list[int]) -> list[dict]:
    """Return all experiment rows for the given model_ids."""
    wanted = set(model_ids)
    return [e for e in _experiments() if e["model_id"] in wanted]


def get_experiment_images(experiment_ids: list[int], only_low_score: bool = False,
                          limit: int = 20) -> list[dict]:
    """Return experiment_images rows for given experiments.

    only_low_score=True filters to top_probability < 0.5 (failure-mode grounding).
    """
    wanted = set(experiment_ids)
    rows = [img for img in _experiment_images() if img["experiment_id"] in wanted]
    if only_low_score:
        rows = [img for img in rows if img["top_probability"] < 0.5]
    # Sort by top_probability ascending so the worst cases come first
    rows.sort(key=lambda img: img["top_probability"])
    return rows[:limit]


def get_device_stats_for_task(category: str, input_type: str | None = None) -> dict:
    """Aggregate device usage statistics across all experiments matching category."""
    index = _index()
    if input_type is not None:
        matching_model_ids = {
            info["model_id"]
            for info in index.values()
            if info["category"] == category and info["input_type"] == input_type
        }
    else:
        matching_model_ids = {
            info["model_id"]
            for info in index.values()
            if info["category"] == category
        }

    exps = [e for e in _experiments() if e["model_id"] in matching_model_ids]
    devices = _devices_by_id()
    stats: dict[str, dict] = {}
    for e in exps:
        dev = devices[e["edge_device_id"]]
        dt = dev["device_type"]
        bucket = stats.setdefault(dt, {
            "device_type": dt, "ram_mb": dev["ram_mb"], "class": dev["class"],
            "run_count": 0, "f1_scores": [], "latencies_ms": [],
        })
        bucket["run_count"] += 1
        bucket["f1_scores"].append(e["f1_score"])
        bucket["latencies_ms"].append(e["per_image_latency_ms"])
    return stats


def get_power_stats(experiment_ids: list[int]) -> list[dict]:
    """Return per-experiment power draw (CPU + GPU watts)."""
    wanted = set(experiment_ids)
    return [
        {"experiment_id": e["id"],
         "total_cpu_power_w": e["total_cpu_power_w"],
         "total_gpu_power_w": e["total_gpu_power_w"],
         "total_power_w": e["total_cpu_power_w"] + e["total_gpu_power_w"]}
        for e in _experiments() if e["id"] in wanted
    ]


def get_card_by_hf_id(hf_card_id: int) -> dict | None:
    """Look up the (category, input_type, model_id) for an HF card by its id."""
    info = _index().get(str(hf_card_id))
    return None if info is None else {**info, "hf_card_id": hf_card_id}
