import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from src.hybrid_schema_matcher import (
        HybridSchemaMatcher,
        LocalOpenAICompatibleLLM,
    )
else:
    from .hybrid_schema_matcher import (
        HybridSchemaMatcher,
        LocalOpenAICompatibleLLM,
    )


@dataclass
class QueryCase:
    query_id: str
    expected_schema_id: str
    target_schema: Dict[str, Any]
    site: str
    split: str
    generator: str
    csv_path: str


@dataclass
class QueryResult:
    query_id: str
    expected_schema_id: str
    predicted_schema_id: str
    exact_match: int
    site: str
    split: str
    generator: str
    csv_path: str
    method: str
    confidence: float
    top_3: List[Dict[str, Any]]


def _sanitize_columns(raw_header: List[str]) -> List[str]:
    seen: Dict[str, int] = {}
    sanitized: List[str] = []
    for index, name in enumerate(raw_header, start=1):
        candidate = (name or "").strip() or f"unnamed_column_{index}"
        if candidate in seen:
            seen[candidate] += 1
            candidate = f"{candidate}_{seen[candidate]}"
        else:
            seen[candidate] = 1
        sanitized.append(candidate)
    return sanitized


def _is_integer_like(value: float) -> bool:
    return float(value).is_integer()


def _percentile(sorted_values: List[float], ratio: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute percentile of an empty list.")
    index = int(round((len(sorted_values) - 1) * ratio))
    index = max(0, min(index, len(sorted_values) - 1))
    return sorted_values[index]


def _load_csv_columns(path: Path) -> Tuple[List[str], Dict[str, List[str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        raw_header = next(reader)
        header = _sanitize_columns(raw_header)
        columns = {name: [] for name in header}
        for row in reader:
            if len(row) < len(header):
                row = row + [""] * (len(header) - len(row))
            for name, value in zip(header, row):
                columns[name].append(value)
    return header, columns


def _build_field_schema(values: List[str]) -> Dict[str, Any]:
    non_empty = [value for value in values if value != ""]
    numeric_values: List[float] = []
    all_numeric = bool(non_empty)
    all_integer = True
    for value in non_empty:
        try:
            numeric_value = float(value)
        except ValueError:
            all_numeric = False
            break
        numeric_values.append(numeric_value)
        if not _is_integer_like(numeric_value):
            all_integer = False

    if all_numeric and numeric_values:
        sorted_values = sorted(numeric_values)
        profile = {
            "min": min(sorted_values),
            "max": max(sorted_values),
            "p50": _percentile(sorted_values, 0.50),
            "p90": _percentile(sorted_values, 0.90),
            "p99": _percentile(sorted_values, 0.99),
            "zero_ratio": round(
                sum(1 for item in sorted_values if item == 0) / len(sorted_values),
                4,
            ),
            "neg_ratio": round(
                sum(1 for item in sorted_values if item < 0) / len(sorted_values),
                4,
            ),
        }
        schema_type = "integer" if all_integer else "number"
        return {
            "type": schema_type,
            "description": "Observed numeric workload column with structured profile metadata.",
            "minimum": profile["min"],
            "maximum": profile["max"],
            "x-profile": profile,
        }

    counts = Counter(non_empty)
    top_values = [[str(value), count] for value, count in counts.most_common(5)]
    field_schema: Dict[str, Any] = {
        "type": "string",
        "description": "Observed categorical workload column with structured profile metadata.",
        "x-profile": {"top_values": top_values},
    }
    return field_schema


def build_schema_from_csv(path: Path) -> Dict[str, Any]:
    header, columns = _load_csv_columns(path)
    properties = {name: _build_field_schema(columns[name]) for name in header}
    required = [name for name in header if all(value != "" for value in columns[name])]
    return {
        "type": "object",
        "description": (
            "HPC parallel workload trace schema derived from observed tabular column profiles."
        ),
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _site_to_training_id(site: str) -> str:
    return f"{site.lower().replace('-', '_')}_training"


def build_benchmark_assets(repo_dir: Path) -> Tuple[List[Dict[str, Any]], List[QueryCase]]:
    schema_records: List[Dict[str, Any]] = []
    queries: List[QueryCase] = []

    for site_dir in sorted(path for path in repo_dir.iterdir() if path.is_dir()):
        site = site_dir.name
        training_dir = site_dir / "training_data"
        generated_dir = site_dir / "generated_data"
        training_files = sorted(training_dir.glob("*.csv")) if training_dir.exists() else []
        if not training_files:
            continue

        training_path = training_files[0]
        training_schema_id = _site_to_training_id(site)
        training_schema = build_schema_from_csv(training_path)
        schema_records.append({"id": training_schema_id, "schema": training_schema})
        queries.append(
            QueryCase(
                query_id=training_schema_id,
                expected_schema_id=training_schema_id,
                target_schema=training_schema,
                site=site,
                split="training",
                generator="training",
                csv_path=str(training_path),
            )
        )

        for generated_path in sorted(generated_dir.glob("*.csv")) if generated_dir.exists() else []:
            generator = generated_path.stem.replace("synthetic_data_", "").split("_")[0]
            queries.append(
                QueryCase(
                    query_id=generated_path.stem,
                    expected_schema_id=training_schema_id,
                    target_schema=build_schema_from_csv(generated_path),
                    site=site,
                    split="generated",
                    generator=generator,
                    csv_path=str(generated_path),
                )
            )

    return schema_records, queries


def export_assets(
    schema_records: List[Dict[str, Any]],
    queries: List[QueryCase],
    asset_dir: Path,
) -> Tuple[Path, Path]:
    asset_dir.mkdir(parents=True, exist_ok=True)
    db_path = asset_dir / "schema_database.json"
    queries_path = asset_dir / "queries.json"
    db_path.write_text(json.dumps(schema_records, indent=2), encoding="utf-8")
    queries_path.write_text(
        json.dumps(
            [
                {
                    "id": query.query_id,
                    "expected_schema_id": query.expected_schema_id,
                    "target_schema": query.target_schema,
                    "meta": {
                        "site": query.site,
                        "split": query.split,
                        "generator": query.generator,
                        "csv_path": query.csv_path,
                    },
                }
                for query in queries
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    return db_path, queries_path


def build_matcher(
    schema_records: List[Dict[str, Any]],
    api_base: str,
    model: str,
    api_key: str,
    timeout: int,
    disable_llm: bool,
) -> HybridSchemaMatcher:
    llm_client = None
    if not disable_llm:
        llm_client = LocalOpenAICompatibleLLM(
            api_base=api_base,
            model=model,
            api_key=api_key,
            timeout_seconds=timeout,
        )
    return HybridSchemaMatcher(schema_records=schema_records, llm_client=llm_client)


def run_method(
    matcher: HybridSchemaMatcher,
    queries: Iterable[QueryCase],
    method: str,
    top_k: int,
) -> List[QueryResult]:
    results: List[QueryResult] = []
    for query in queries:
        if method == "hybrid":
            result = matcher.match_schema(query.target_schema, top_k=top_k)
        elif method == "pure_cosine":
            result = matcher.pure_cosine_match_schema(query.target_schema, top_k=top_k)
        else:
            raise ValueError(f"Unsupported method: {method}")

        results.append(
            QueryResult(
                query_id=query.query_id,
                expected_schema_id=query.expected_schema_id,
                predicted_schema_id=result.winner,
                exact_match=int(result.winner == query.expected_schema_id),
                site=query.site,
                split=query.split,
                generator=query.generator,
                csv_path=query.csv_path,
                method=result.method,
                confidence=float(result.confidence),
                top_3=result.report.get("top_3", []),
            )
        )
    return results


def _accuracy(rows: List[QueryResult]) -> float:
    if not rows:
        return 0.0
    return sum(row.exact_match for row in rows) / len(rows)


def _rows_by_key(rows: Iterable[QueryResult], key: str) -> Dict[str, List[QueryResult]]:
    grouped: Dict[str, List[QueryResult]] = defaultdict(list)
    for row in rows:
        grouped[str(getattr(row, key))].append(row)
    return dict(grouped)


def summarize_results(rows: List[QueryResult]) -> Dict[str, Any]:
    grouped_by_site = _rows_by_key(rows, "site")
    grouped_by_generator = _rows_by_key(rows, "generator")
    overall = {
        "trace_source_accuracy_at_1": _accuracy(rows),
        "generated_trace_source_accuracy_at_1": _accuracy(
            [row for row in rows if row.split == "generated"]
        ),
        "training_self_accuracy_at_1": _accuracy(
            [row for row in rows if row.split == "training"]
        ),
    }
    return {
        "aggregate": overall,
        "by_site": {
            site: {
                "count": len(site_rows),
                "trace_source_accuracy_at_1": _accuracy(site_rows),
            }
            for site, site_rows in sorted(grouped_by_site.items())
        },
        "by_generator": {
            generator: {
                "count": len(generator_rows),
                "trace_source_accuracy_at_1": _accuracy(generator_rows),
            }
            for generator, generator_rows in sorted(grouped_by_generator.items())
        },
        "per_query": [asdict(row) for row in rows],
    }


def build_delta(hybrid: Dict[str, Any], pure_cosine: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "trace_source_accuracy_at_1": (
            hybrid["aggregate"]["trace_source_accuracy_at_1"]
            - pure_cosine["aggregate"]["trace_source_accuracy_at_1"]
        ),
        "generated_trace_source_accuracy_at_1": (
            hybrid["aggregate"]["generated_trace_source_accuracy_at_1"]
            - pure_cosine["aggregate"]["generated_trace_source_accuracy_at_1"]
        ),
        "training_self_accuracy_at_1": (
            hybrid["aggregate"]["training_self_accuracy_at_1"]
            - pure_cosine["aggregate"]["training_self_accuracy_at_1"]
        ),
    }


def _format_metric(value: float) -> str:
    return f"{value:.4f}"


def _failure_rows(
    hybrid_rows: List[Dict[str, Any]],
    cosine_rows: List[Dict[str, Any]],
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    cosine_by_id = {row["query_id"]: row for row in cosine_rows}
    failures: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for hybrid_row in hybrid_rows:
        cosine_row = cosine_by_id[hybrid_row["query_id"]]
        if hybrid_row["exact_match"] == 1 and cosine_row["exact_match"] == 0:
            failures.append((hybrid_row, cosine_row))
    return failures


def render_markdown_report(results: Dict[str, Any]) -> str:
    hybrid = results["methods"]["hybrid"]
    pure_cosine = results["methods"]["pure_cosine"]
    delta = results["delta"]
    failures = _failure_rows(hybrid["per_query"], pure_cosine["per_query"])

    lines = [
        "# Gen-Parallel-Workloads Benchmark",
        "",
        "## Setup",
        "",
        f"- Repository: `{results['repo_dir']}`",
        f"- Exported database: `{results['db_path']}`",
        f"- Exported queries: `{results['queries_path']}`",
        f"- Training candidates: `{results['training_candidates']}`",
        f"- Evaluation queries: `{results['query_count']}`",
        f"- Generated queries: `{results['generated_query_count']}`",
        f"- LLM enabled: `{str(results['llm_enabled']).lower()}`",
        f"- Hybrid model: `{results['model']}`",
        "",
        "## Metric",
        "",
        "`Trace Source Accuracy@1 (TSA@1)` measures whether the top-1 retrieved training trace comes from the same source workload as the query trace. This benchmark needs source retrieval rather than pure structural metrics because BW, Helios, Philly, and Theta share the same column names but differ sharply in value distributions.",
        "",
        "## Aggregate Accuracy",
        "",
        "| Method | TSA@1 | Generated TSA@1 | Training Self-Accuracy@1 |",
        "| --- | ---: | ---: | ---: |",
        f"| Hybrid | {_format_metric(hybrid['aggregate']['trace_source_accuracy_at_1'])} | {_format_metric(hybrid['aggregate']['generated_trace_source_accuracy_at_1'])} | {_format_metric(hybrid['aggregate']['training_self_accuracy_at_1'])} |",
        f"| Pure cosine | {_format_metric(pure_cosine['aggregate']['trace_source_accuracy_at_1'])} | {_format_metric(pure_cosine['aggregate']['generated_trace_source_accuracy_at_1'])} | {_format_metric(pure_cosine['aggregate']['training_self_accuracy_at_1'])} |",
        "",
        "## Delta (Hybrid - Pure Cosine)",
        "",
        "| Metric | Delta |",
        "| --- | ---: |",
        f"| TSA@1 | {_format_metric(delta['trace_source_accuracy_at_1'])} |",
        f"| Generated TSA@1 | {_format_metric(delta['generated_trace_source_accuracy_at_1'])} |",
        f"| Training Self-Accuracy@1 | {_format_metric(delta['training_self_accuracy_at_1'])} |",
        "",
        "## Accuracy By Site",
        "",
        "| Site | Queries | Hybrid TSA@1 | Pure cosine TSA@1 |",
        "| --- | ---: | ---: | ---: |",
    ]

    for site in sorted(hybrid["by_site"]):
        hybrid_site = hybrid["by_site"][site]
        cosine_site = pure_cosine["by_site"][site]
        lines.append(
            f"| {site} | {hybrid_site['count']} | "
            f"{_format_metric(hybrid_site['trace_source_accuracy_at_1'])} | "
            f"{_format_metric(cosine_site['trace_source_accuracy_at_1'])} |"
        )

    lines.extend(
        [
            "",
            "## Accuracy By Generator",
            "",
            "| Generator | Queries | Hybrid TSA@1 | Pure cosine TSA@1 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for generator in sorted(hybrid["by_generator"]):
        hybrid_generator = hybrid["by_generator"][generator]
        cosine_generator = pure_cosine["by_generator"][generator]
        lines.append(
            f"| {generator} | {hybrid_generator['count']} | "
            f"{_format_metric(hybrid_generator['trace_source_accuracy_at_1'])} | "
            f"{_format_metric(cosine_generator['trace_source_accuracy_at_1'])} |"
        )

    lines.extend(
        [
            "",
            "## Failure Cases Fixed By Hybrid",
            "",
            "| Query | Site | Expected | Hybrid Top-1 | Cosine Top-1 | Cosine Confidence |",
            "| --- | --- | --- | --- | --- | ---: |",
        ]
    )
    for hybrid_row, cosine_row in failures:
        lines.append(
            f"| {hybrid_row['query_id']} | {hybrid_row['site']} | {hybrid_row['expected_schema_id']} | "
            f"{hybrid_row['predicted_schema_id']} | {cosine_row['predicted_schema_id']} | "
            f"{_format_metric(cosine_row['confidence'])} |"
        )

    if not failures:
        lines.append("| none | - | - | - | - | - |")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the hybrid schema matcher on the Gen-Parallel-Workloads CSV repository."
    )
    parser.add_argument(
        "--repo-dir",
        default="external/Gen-Parallel-Workloads",
        help="Path to the cloned Gen-Parallel-Workloads repository.",
    )
    parser.add_argument(
        "--asset-dir",
        default="examples/gen_parallel_workloads_benchmark",
        help="Directory where derived schema/queries JSON assets should be written.",
    )
    parser.add_argument(
        "--json-out",
        default="reports/gen_parallel_workloads_report.json",
        help="Path to write the raw benchmark results JSON.",
    )
    parser.add_argument(
        "--markdown-out",
        default="reports/gen_parallel_workloads_report.md",
        help="Path to write the Markdown benchmark report.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of candidates to recall before hybrid reranking.",
    )
    parser.add_argument(
        "--api-base",
        default="http://127.0.0.1:1234/v1",
        help="OpenAI-compatible API base URL for the hybrid reranker.",
    )
    parser.add_argument(
        "--model",
        default="qwen35-9b",
        help="Model name exposed by the OpenAI-compatible API.",
    )
    parser.add_argument(
        "--api-key",
        default="lm-studio",
        help="API key for the OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="LLM request timeout in seconds.",
    )
    parser.add_argument(
        "--disable-llm",
        action="store_true",
        help="Disable the LLM reranker and use deterministic fallback for the hybrid method.",
    )

    args = parser.parse_args()
    repo_dir = Path(args.repo_dir)
    asset_dir = Path(args.asset_dir)
    json_out = Path(args.json_out)
    markdown_out = Path(args.markdown_out)

    schema_records, queries = build_benchmark_assets(repo_dir)
    db_path, queries_path = export_assets(schema_records, queries, asset_dir)
    matcher = build_matcher(
        schema_records=schema_records,
        api_base=args.api_base,
        model=args.model,
        api_key=args.api_key,
        timeout=args.timeout,
        disable_llm=args.disable_llm,
    )

    hybrid_rows = run_method(matcher, queries, method="hybrid", top_k=args.top_k)
    pure_cosine_rows = run_method(matcher, queries, method="pure_cosine", top_k=args.top_k)

    output = {
        "repo_dir": str(repo_dir),
        "db_path": str(db_path),
        "queries_path": str(queries_path),
        "model": args.model,
        "llm_enabled": not args.disable_llm,
        "training_candidates": len(schema_records),
        "query_count": len(queries),
        "generated_query_count": sum(1 for query in queries if query.split == "generated"),
        "methods": {
            "hybrid": summarize_results(hybrid_rows),
            "pure_cosine": summarize_results(pure_cosine_rows),
        },
    }
    output["delta"] = build_delta(output["methods"]["hybrid"], output["methods"]["pure_cosine"])

    json_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    markdown_out.write_text(render_markdown_report(output), encoding="utf-8")

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
