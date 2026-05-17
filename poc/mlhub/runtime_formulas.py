"""Deterministic aggregation formulas for the runtime augmentation pipeline.

Single source of truth shared by:
  - build_mock_experiments.py (to derive ground-truth per card)
  - augment_runtime.py       (to compute runtime field values at augment time)

Because both callers import the same functions, the oracle cannot drift from
the augmentation output — which is the promise M4 makes to judge evaluation.

All functions operate on a list of experiment rows (the shape emitted by the
fixture's `experiments[]`, which mirrors the Patra `experiments` table plus
`total_cpu_power_w`/`total_gpu_power_w` from the power-summary stream).

Device metadata is provided by the caller via `devices_by_id`. This keeps the
module free of fixture-loading and lets tests inject their own device table.
"""

from __future__ import annotations

import statistics


def suggested_hardware(exps: list[dict], devices_by_id: dict[int, dict],
                       valid_device_types: set[str] | None = None) -> str | None:
    """Argmax(device_type) over similar-model experiments, alphabetical tiebreak.

    If valid_device_types is given, the return value must be in the set; any
    unknown device is coerced to None. This prevents drift ("A100" vs
    "nvidia-a100") when the source data is noisy.
    """
    if not exps:
        return None
    counts: dict[str, int] = {}
    for e in exps:
        dt = devices_by_id[e["edge_device_id"]]["device_type"]
        counts[dt] = counts.get(dt, 0) + 1
    winner = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    if valid_device_types is not None and winner not in valid_device_types:
        return None
    return winner


def expected_f1_range(exps: list[dict]) -> list[float] | None:
    """[p25, p75] of f1_score, linear interpolation. Needs ≥2 experiments."""
    if len(exps) < 2:
        return None
    f1s = sorted(e["f1_score"] for e in exps)
    qs = statistics.quantiles(f1s, n=4, method="inclusive")
    return [round(qs[0], 5), round(qs[2], 5)]


def expected_latency_ms(exps: list[dict]) -> float | None:
    """Median per-image latency in ms."""
    if not exps:
        return None
    return round(statistics.median(e["per_image_latency_ms"] for e in exps), 2)


def deployment_maturity(exps: list[dict]) -> str | None:
    """experimental | validated | production — by distinct devices × run count."""
    if not exps:
        return None
    distinct_devices = len({e["edge_device_id"] for e in exps})
    n_runs = len(exps)
    if distinct_devices >= 3 and n_runs >= 10:
        return "production"
    if distinct_devices >= 2 and n_runs >= 5:
        return "validated"
    return "experimental"


def recommended_min_ram_mb(exps: list[dict], devices_by_id: dict[int, dict]) -> int | None:
    """Min RAM among devices where this model ran with f1 ≥ 0.6."""
    candidates = [devices_by_id[e["edge_device_id"]]["ram_mb"]
                  for e in exps if e["f1_score"] >= 0.6]
    return min(candidates) if candidates else None


def inference_cost_class(exps: list[dict], devices_by_id: dict[int, dict]) -> str | None:
    """low | medium | high — by dominant device class + median latency."""
    if not exps:
        return None
    median_lat = expected_latency_ms(exps)
    class_counts: dict[str, int] = {}
    for e in exps:
        cls = devices_by_id[e["edge_device_id"]]["class"]
        class_counts[cls] = class_counts.get(cls, 0) + 1
    dom_class = sorted(class_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    if dom_class == "datacenter":
        return "high"
    if dom_class == "cpu" or (dom_class == "edge" and median_lat is not None and median_lat > 50):
        return "medium"
    return "low"


def expected_total_power_w(exps: list[dict]) -> float | None:
    """Median (total_cpu_power_w + total_gpu_power_w) across experiments, in watts."""
    if not exps:
        return None
    return round(statistics.median(e["total_cpu_power_w"] + e["total_gpu_power_w"] for e in exps), 3)


# ---------------------------------------------------------------------------
# MLHub helpers (p95-style aggregations for ModelMetadata deployment fields).
# Nearest-rank p95: for small samples this converges to max(), which is the
# conservative interpretation for "max_*" deployment limits.
# ---------------------------------------------------------------------------

import math


def _nearest_rank(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile. p ∈ (0, 1]."""
    if not values:
        return None
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(math.ceil(p * len(s))) - 1))
    return s[k]


def _derive_per_image_latency_ms(exp: dict) -> float | None:
    """Prefer an explicit per_image_latency_ms; otherwise derive from timestamps."""
    if exp.get("per_image_latency_ms") is not None:
        return float(exp["per_image_latency_ms"])
    ts_start = exp.get("image_receiving_timestamp")
    ts_end = exp.get("image_scoring_timestamp")
    n_imgs = exp.get("total_images")
    if not ts_start or not ts_end or not n_imgs:
        return None
    from datetime import datetime
    try:
        t0 = datetime.fromisoformat(ts_start.replace("Z", "+00:00"))
        t1 = datetime.fromisoformat(ts_end.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None
    delta_ms = (t1 - t0).total_seconds() * 1000.0
    if n_imgs <= 0 or delta_ms <= 0:
        return None
    return delta_ms / n_imgs


def p95_latency_ms(exps: list[dict]) -> float | None:
    """p95 of per-image latency across experiments (nearest-rank); robust to tail noise."""
    if not exps:
        return None
    lats: list[float] = []
    for e in exps:
        v = _derive_per_image_latency_ms(e)
        if v is not None:
            lats.append(v)
    if not lats:
        return None
    v = _nearest_rank(lats, 0.95)
    return round(v, 2) if v is not None else None


def p95_total_power_w(exps: list[dict]) -> float | None:
    """p95 of (total_cpu_power_w + total_gpu_power_w) across experiments."""
    if not exps:
        return None
    totals = [float(e["total_cpu_power_w"]) + float(e["total_gpu_power_w"])
              for e in exps
              if e.get("total_cpu_power_w") is not None and e.get("total_gpu_power_w") is not None]
    if not totals:
        return None
    v = _nearest_rank(totals, 0.95)
    return round(v, 3) if v is not None else None


def min_throughput(exps: list[dict]) -> float | None:
    """floor(1000 / p95_latency_ms) — minimum images/sec based on worst-case latency."""
    p95 = p95_latency_ms(exps)
    if p95 is None or p95 <= 0:
        return None
    return math.floor(1000.0 / p95)


def any_distributed(exps: list[dict]) -> bool | None:
    """Whether inference ran distributed across multiple devices.

    Each experiment row in the fixture references a single `edge_device_id`,
    so a single experiment is never itself distributed. Returns False when
    experiments exist (signalling "no distributed runs observed"), None when
    the list is empty.
    """
    if not exps:
        return None
    return False


def dominant_device(exps: list[dict], devices_by_id: dict[int, dict]) -> dict | None:
    """Return the full device row for the device_type with the most experiments.

    Tiebreak: alphabetical on device_type.
    """
    if not exps:
        return None
    counts: dict[int, int] = {}
    for e in exps:
        counts[e["edge_device_id"]] = counts.get(e["edge_device_id"], 0) + 1
    # Sort by (-count, device_type alphabetical) — same tiebreak as suggested_hardware
    best_id = sorted(counts.keys(),
                     key=lambda did: (-counts[did], devices_by_id[did]["device_type"]))[0]
    return devices_by_id.get(best_id)

