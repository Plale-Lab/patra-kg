#!/usr/bin/env python3
"""Smoke tests for the runtime formula module.

Pins each of the 7 aggregations to specific inputs -> specific outputs. If
someone edits runtime_formulas.py, these tests catch drift instantly.

Run:
    python poc/test_runtime_formulas.py

Exit code 0 on success; any failure prints the mismatch and exits 1.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import runtime_formulas as rf

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DEVICES = {
    1: {"device_type": "Jetson Nano", "ram_mb": 4096,  "class": "edge"},
    2: {"device_type": "A100",        "ram_mb": 81920, "class": "datacenter"},
    3: {"device_type": "CPU-x86",     "ram_mb": 16384, "class": "cpu"},
}
VALID_TYPES = {d["device_type"] for d in DEVICES.values()}


def exp(device_id: int, f1: float, latency_ms: int = 30,
        cpu_w: float = 5.0, gpu_w: float = 0.0) -> dict:
    """Build a minimal experiment row."""
    return {
        "edge_device_id": device_id,
        "f1_score": f1,
        "per_image_latency_ms": latency_ms,
        "total_cpu_power_w": cpu_w,
        "total_gpu_power_w": gpu_w,
    }


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

_failures: list[str] = []


def check(label: str, actual, expected) -> None:
    if actual == expected:
        print(f"  OK   {label}")
    else:
        _failures.append(f"{label}\n    expected: {expected!r}\n    actual:   {actual!r}")
        print(f"  FAIL {label}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_suggested_hardware() -> None:
    print("\nsuggested_hardware")
    # Empty -> None
    check("empty list -> None", rf.suggested_hardware([], DEVICES, VALID_TYPES), None)
    # Single experiment
    check("single A100 run -> A100", rf.suggested_hardware([exp(2, 0.9)], DEVICES, VALID_TYPES), "A100")
    # Argmax: 2× Jetson, 1× A100 -> Jetson
    check("2x Jetson + 1x A100 -> Jetson Nano",
          rf.suggested_hardware([exp(1, 0.9), exp(1, 0.9), exp(2, 0.9)], DEVICES, VALID_TYPES),
          "Jetson Nano")
    # Tie: alphabetical. A100 < CPU-x86 alphabetically.
    check("tie A100 vs CPU-x86 -> A100 (alphabetical)",
          rf.suggested_hardware([exp(2, 0.9), exp(3, 0.9)], DEVICES, VALID_TYPES),
          "A100")
    # Tie: CPU-x86 vs Jetson Nano -> CPU-x86 (alphabetical)
    check("tie CPU-x86 vs Jetson Nano -> CPU-x86",
          rf.suggested_hardware([exp(1, 0.9), exp(3, 0.9)], DEVICES, VALID_TYPES),
          "CPU-x86")
    # Enum constraint: unknown device type nulls the result
    bogus_devices = {9: {"device_type": "SomethingWeird", "ram_mb": 0, "class": "edge"}}
    check("unknown device with enum -> None",
          rf.suggested_hardware([exp(9, 0.9)], bogus_devices, VALID_TYPES),
          None)
    # No enum constraint -> passthrough
    check("unknown device, no enum -> passthrough",
          rf.suggested_hardware([exp(9, 0.9)], bogus_devices, None),
          "SomethingWeird")


def test_expected_f1_range() -> None:
    print("\nexpected_f1_range")
    check("empty -> None", rf.expected_f1_range([]), None)
    check("single experiment -> None", rf.expected_f1_range([exp(1, 0.8)]), None)
    # Two experiments: inclusive quantiles interpolate linearly between the two points.
    # [0.6, 0.8] -> p25 = 0.65, p75 = 0.75.
    check("2 experiments [0.6, 0.8] -> [0.65, 0.75]",
          rf.expected_f1_range([exp(1, 0.6), exp(1, 0.8)]), [0.65, 0.75])
    # Five f1s: [0.1, 0.2, 0.3, 0.4, 0.5], inclusive quantiles at n=4 -> p25=0.2, p75=0.4
    f1s = [0.1, 0.2, 0.3, 0.4, 0.5]
    check("5 experiments p25/p75",
          rf.expected_f1_range([exp(1, v) for v in f1s]),
          [0.2, 0.4])


def test_expected_latency_ms() -> None:
    print("\nexpected_latency_ms")
    check("empty -> None", rf.expected_latency_ms([]), None)
    check("single 30ms -> 30", rf.expected_latency_ms([exp(1, 0.9, latency_ms=30)]), 30)
    # Median of [10, 20, 30] = 20
    check("median [10, 20, 30] = 20",
          rf.expected_latency_ms([exp(1, 0.9, latency_ms=n) for n in [10, 20, 30]]), 20)
    # Median of [10, 20] = 15.0 (rounded to 15 from statistics.median of two ints)
    check("median [10, 20] = 15",
          rf.expected_latency_ms([exp(1, 0.9, latency_ms=n) for n in [10, 20]]), 15)


def test_deployment_maturity() -> None:
    print("\ndeployment_maturity")
    check("empty -> None", rf.deployment_maturity([]), None)
    # 1 device, 1 run -> experimental
    check("1 device × 1 run -> experimental",
          rf.deployment_maturity([exp(1, 0.9)]), "experimental")
    # 2 devices, 5 runs -> validated
    runs = [exp(1, 0.9), exp(1, 0.9), exp(1, 0.9), exp(2, 0.9), exp(2, 0.9)]
    check("2 devices × 5 runs -> validated",
          rf.deployment_maturity(runs), "validated")
    # 2 devices × 4 runs -> experimental (below 5-run bar)
    runs_4 = [exp(1, 0.9), exp(1, 0.9), exp(2, 0.9), exp(2, 0.9)]
    check("2 devices × 4 runs -> experimental",
          rf.deployment_maturity(runs_4), "experimental")
    # 3 devices, 10 runs -> production
    runs_10 = [exp(1, 0.9)] * 4 + [exp(2, 0.9)] * 3 + [exp(3, 0.9)] * 3
    check("3 devices × 10 runs -> production",
          rf.deployment_maturity(runs_10), "production")
    # 3 devices, 9 runs -> validated (below 10-run bar)
    runs_9 = [exp(1, 0.9)] * 3 + [exp(2, 0.9)] * 3 + [exp(3, 0.9)] * 3
    check("3 devices × 9 runs -> validated",
          rf.deployment_maturity(runs_9), "validated")


def test_recommended_min_ram_mb() -> None:
    print("\nrecommended_min_ram_mb")
    check("empty -> None", rf.recommended_min_ram_mb([], DEVICES), None)
    # All below 0.6 f1 -> None
    check("all f1 < 0.6 -> None",
          rf.recommended_min_ram_mb([exp(1, 0.3), exp(2, 0.5)], DEVICES),
          None)
    # Jetson (4096) passes f1=0.7 -> 4096
    check("Jetson @ f1=0.7 -> 4096",
          rf.recommended_min_ram_mb([exp(1, 0.7), exp(2, 0.8)], DEVICES),
          4096)
    # Only A100 passes (Jetson at f1=0.5 fails) -> 81920
    check("only A100 passes (Jetson fails f1 bar) -> 81920",
          rf.recommended_min_ram_mb([exp(1, 0.5), exp(2, 0.8)], DEVICES),
          81920)


def test_inference_cost_class() -> None:
    print("\ninference_cost_class")
    check("empty -> None", rf.inference_cost_class([], DEVICES), None)
    # All datacenter -> high
    check("all A100 -> high",
          rf.inference_cost_class([exp(2, 0.9)] * 3, DEVICES), "high")
    # All CPU -> medium
    check("all CPU-x86 -> medium",
          rf.inference_cost_class([exp(3, 0.9)] * 3, DEVICES), "medium")
    # All edge, median latency 30ms (<=50) -> low
    check("all Jetson fast -> low",
          rf.inference_cost_class([exp(1, 0.9, latency_ms=30)] * 3, DEVICES), "low")
    # All edge, median latency 80ms (>50) -> medium
    check("all Jetson slow -> medium",
          rf.inference_cost_class([exp(1, 0.9, latency_ms=80)] * 3, DEVICES), "medium")
    # Majority datacenter -> high (tie-breaks handled by alphabetical class ordering)
    runs = [exp(2, 0.9), exp(2, 0.9), exp(1, 0.9)]
    check("2 A100 + 1 Jetson -> high (datacenter dominant)",
          rf.inference_cost_class(runs, DEVICES), "high")


def test_expected_total_power_w() -> None:
    print("\nexpected_total_power_w")
    check("empty -> None", rf.expected_total_power_w([]), None)
    # Single Jetson: 3 + 5 = 8.0
    check("single Jetson (3 + 5) = 8.0",
          rf.expected_total_power_w([exp(1, 0.9, cpu_w=3.0, gpu_w=5.0)]), 8.0)
    # Median of three experiments with totals 10, 20, 30 -> 20
    runs = [
        exp(1, 0.9, cpu_w=5.0,  gpu_w=5.0),    # 10
        exp(1, 0.9, cpu_w=10.0, gpu_w=10.0),   # 20
        exp(1, 0.9, cpu_w=15.0, gpu_w=15.0),   # 30
    ]
    check("median [10, 20, 30] -> 20",
          rf.expected_total_power_w(runs), 20.0)


# ---------------------------------------------------------------------------
# MLHub helpers
# ---------------------------------------------------------------------------

def test_p95_latency_ms() -> None:
    print("\np95_latency_ms")
    check("empty -> None", rf.p95_latency_ms([]), None)
    # Single 30ms -> 30 (nearest-rank p95 on n=1 is the value itself)
    check("single 30ms -> 30.0",
          rf.p95_latency_ms([exp(1, 0.9, latency_ms=30)]), 30.0)
    # nearest-rank p95 on n=10: ceil(0.95 * 10) - 1 = 9 -> sorted[9] = max = 100
    lats = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    check("p95 of 10 values -> 100.0",
          rf.p95_latency_ms([exp(1, 0.9, latency_ms=v) for v in lats]), 100.0)
    # With n=20: ceil(0.95 * 20) - 1 = 18 -> sorted[18] = 95 (not the max)
    lats20 = list(range(5, 105, 5))  # [5, 10, ..., 100]
    check("p95 of 20 values -> 95.0",
          rf.p95_latency_ms([exp(1, 0.9, latency_ms=v) for v in lats20]), 95.0)


def test_p95_total_power_w() -> None:
    print("\np95_total_power_w")
    check("empty -> None", rf.p95_total_power_w([]), None)
    # Single Jetson: 3 + 5 = 8.0
    check("single Jetson (3 + 5) = 8.0",
          rf.p95_total_power_w([exp(1, 0.9, cpu_w=3.0, gpu_w=5.0)]), 8.0)
    # 10 values -> nearest-rank p95 is the max (index 9)
    runs = [exp(1, 0.9, cpu_w=float(v), gpu_w=0.0) for v in range(1, 11)]
    check("p95 of [1..10] -> 10.0",
          rf.p95_total_power_w(runs), 10.0)


def test_min_throughput() -> None:
    print("\nmin_throughput")
    check("empty -> None", rf.min_throughput([]), None)
    # p95 latency = 100ms -> floor(1000 / 100) = 10
    lats = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    check("p95 lat 100ms -> 10 img/s",
          rf.min_throughput([exp(1, 0.9, latency_ms=v) for v in lats]), 10)
    # Single 50ms -> 1000 / 50 = 20
    check("p95 lat 50ms -> 20 img/s",
          rf.min_throughput([exp(1, 0.9, latency_ms=50)]), 20)


def test_any_distributed() -> None:
    print("\nany_distributed")
    check("empty -> None", rf.any_distributed([]), None)
    check("any runs -> False (single-device per row)",
          rf.any_distributed([exp(1, 0.9)]), False)
    check("multiple devices, still False",
          rf.any_distributed([exp(1, 0.9), exp(2, 0.9)]), False)


def test_dominant_device() -> None:
    print("\ndominant_device")
    check("empty -> None", rf.dominant_device([], DEVICES), None)
    # Single A100 run -> A100 device row
    a100_row = rf.dominant_device([exp(2, 0.9)], DEVICES)
    check("single A100 -> ram_mb=81920", a100_row["ram_mb"], 81920)
    # Argmax: 2x Jetson, 1x A100 -> Jetson Nano
    runs = [exp(1, 0.9), exp(1, 0.9), exp(2, 0.9)]
    jetson = rf.dominant_device(runs, DEVICES)
    check("2x Jetson + 1x A100 -> Jetson Nano", jetson["device_type"], "Jetson Nano")
    check("2x Jetson + 1x A100 -> ram 4096", jetson["ram_mb"], 4096)
    # Tie: A100 vs CPU-x86 -> A100 (alphabetical)
    tied = rf.dominant_device([exp(2, 0.9), exp(3, 0.9)], DEVICES)
    check("tie A100 vs CPU -> A100", tied["device_type"], "A100")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    tests = [
        test_suggested_hardware,
        test_expected_f1_range,
        test_expected_latency_ms,
        test_deployment_maturity,
        test_recommended_min_ram_mb,
        test_inference_cost_class,
        test_expected_total_power_w,
        test_p95_latency_ms,
        test_p95_total_power_w,
        test_min_throughput,
        test_any_distributed,
        test_dominant_device,
    ]
    for t in tests:
        t()

    print()
    if _failures:
        print(f"{len(_failures)} failure(s):")
        for f in _failures:
            print(f"  * {f}")
        return 1
    print("All formula tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
