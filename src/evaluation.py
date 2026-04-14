import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from src.hybrid_schema_matcher import (
        FlattenedField,
        HybridSchemaMatcher,
        LocalOpenAICompatibleLLM,
        MatchResult,
        SchemaDocument,
        load_schema_records,
    )
else:
    from .hybrid_schema_matcher import (
        FlattenedField,
        HybridSchemaMatcher,
        LocalOpenAICompatibleLLM,
        MatchResult,
        SchemaDocument,
        load_schema_records,
    )


@dataclass
class StructuralMetrics:
    field_recall: float
    type_accuracy: float
    required_accuracy: float
    format_accuracy: float
    struct_score: float
    strict_win: int


@dataclass
class QueryEvaluation:
    query_id: str
    expected_schema_id: str
    predicted_schema_id: str
    exact_match: int
    metrics: StructuralMetrics
    method: str
    confidence: float


@dataclass
class AggregateMetrics:
    sma_at_1: float
    strict_win_rate_at_1: float
    top1_exact_accuracy: float
    avg_field_recall: float
    avg_type_accuracy: float
    avg_required_accuracy: float
    avg_format_accuracy: float


def _leaf_fields(document: SchemaDocument) -> Dict[str, FlattenedField]:
    return {
        field.path: field
        for field in document.fields
        if field.json_type not in {"object", "array"} and field.path != "$"
    }


def compute_structural_metrics(
    target_doc: SchemaDocument,
    predicted_doc: SchemaDocument,
) -> StructuralMetrics:
    target_fields = _leaf_fields(target_doc)
    predicted_fields = _leaf_fields(predicted_doc)

    target_paths = set(target_fields)
    predicted_paths = set(predicted_fields)
    matched_paths = sorted(target_paths & predicted_paths)

    field_recall = len(matched_paths) / max(len(target_paths), 1)

    if matched_paths:
        type_matches = 0
        required_matches = 0
        format_matches = 0
        format_conflicts = 0
        for path in matched_paths:
            target_field = target_fields[path]
            predicted_field = predicted_fields[path]
            if target_field.json_type == predicted_field.json_type:
                type_matches += 1
            if target_field.required == predicted_field.required:
                required_matches += 1
            if target_field.fmt == predicted_field.fmt:
                format_matches += 1
            elif target_field.fmt or predicted_field.fmt:
                format_conflicts += 1

        matched_count = len(matched_paths)
        type_accuracy = type_matches / matched_count
        required_accuracy = required_matches / matched_count
        format_accuracy = format_matches / matched_count
    else:
        type_accuracy = 0.0
        required_accuracy = 0.0
        format_accuracy = 0.0
        format_conflicts = 0

    required_paths = {path for path, field in target_fields.items() if field.required}
    required_covered = required_paths.issubset(predicted_paths)
    no_type_conflicts = all(
        target_fields[path].json_type == predicted_fields[path].json_type
        for path in matched_paths
    ) and len(matched_paths) == len(predicted_paths & target_paths)
    strict_win = int(required_covered and no_type_conflicts and format_conflicts == 0)

    struct_score = (
        0.4 * field_recall
        + 0.3 * type_accuracy
        + 0.2 * required_accuracy
        + 0.1 * format_accuracy
    )

    return StructuralMetrics(
        field_recall=field_recall,
        type_accuracy=type_accuracy,
        required_accuracy=required_accuracy,
        format_accuracy=format_accuracy,
        struct_score=struct_score,
        strict_win=strict_win,
    )


def aggregate_query_metrics(rows: List[QueryEvaluation]) -> AggregateMetrics:
    return AggregateMetrics(
        sma_at_1=mean(row.metrics.struct_score for row in rows),
        strict_win_rate_at_1=mean(row.metrics.strict_win for row in rows),
        top1_exact_accuracy=mean(row.exact_match for row in rows),
        avg_field_recall=mean(row.metrics.field_recall for row in rows),
        avg_type_accuracy=mean(row.metrics.type_accuracy for row in rows),
        avg_required_accuracy=mean(row.metrics.required_accuracy for row in rows),
        avg_format_accuracy=mean(row.metrics.format_accuracy for row in rows),
    )


def evaluate_queries(
    matcher: HybridSchemaMatcher,
    queries: List[Dict[str, Any]],
    method: str,
    top_k: int,
) -> Dict[str, Any]:
    rows: List[QueryEvaluation] = []
    for query in queries:
        target_schema = query["target_schema"]
        expected_schema_id = query["expected_schema_id"]
        if method == "hybrid":
            result = matcher.match_schema(target_schema, top_k=top_k)
        elif method == "pure_cosine":
            result = matcher.pure_cosine_match_schema(target_schema, top_k=top_k)
        else:
            raise ValueError(f"Unsupported evaluation method: {method}")

        target_doc = matcher.get_document(expected_schema_id)
        predicted_doc = matcher.get_document(result.winner)
        metrics = compute_structural_metrics(target_doc, predicted_doc)
        rows.append(
            QueryEvaluation(
                query_id=query["id"],
                expected_schema_id=expected_schema_id,
                predicted_schema_id=result.winner,
                exact_match=int(result.winner == expected_schema_id),
                metrics=metrics,
                method=result.method,
                confidence=result.confidence,
            )
        )

    return {
        "aggregate": asdict(aggregate_query_metrics(rows)),
        "per_query": [asdict(row) for row in rows],
    }


def load_queries(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _format_metric(value: float) -> str:
    return f"{value:.4f}"


def render_markdown_report(results: Dict[str, Any]) -> str:
    hybrid = results["methods"]["hybrid"]
    pure_cosine = results["methods"]["pure_cosine"]
    delta = results["delta"]

    lines = [
        "# Evaluation Report",
        "",
        f"- Database: `{results['db']}`",
        f"- Queries: `{results['queries']}`",
        "",
        "## Aggregate Metrics",
        "",
        "| Method | SMA@1 | Strict Win Rate@1 | Top-1 Exact Accuracy | Avg Field Recall | Avg Type Accuracy | Avg Required Accuracy | Avg Format Accuracy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| Hybrid | {_format_metric(hybrid['sma_at_1'])} | {_format_metric(hybrid['strict_win_rate_at_1'])} | {_format_metric(hybrid['top1_exact_accuracy'])} | {_format_metric(hybrid['avg_field_recall'])} | {_format_metric(hybrid['avg_type_accuracy'])} | {_format_metric(hybrid['avg_required_accuracy'])} | {_format_metric(hybrid['avg_format_accuracy'])} |",
        f"| Pure cosine | {_format_metric(pure_cosine['sma_at_1'])} | {_format_metric(pure_cosine['strict_win_rate_at_1'])} | {_format_metric(pure_cosine['top1_exact_accuracy'])} | {_format_metric(pure_cosine['avg_field_recall'])} | {_format_metric(pure_cosine['avg_type_accuracy'])} | {_format_metric(pure_cosine['avg_required_accuracy'])} | {_format_metric(pure_cosine['avg_format_accuracy'])} |",
        "",
        "## Delta (Hybrid - Pure Cosine)",
        "",
        "| Metric | Delta |",
        "| --- | ---: |",
        f"| SMA@1 | {_format_metric(delta['sma_at_1'])} |",
        f"| Strict Win Rate@1 | {_format_metric(delta['strict_win_rate_at_1'])} |",
        f"| Top-1 Exact Accuracy | {_format_metric(delta['top1_exact_accuracy'])} |",
        "",
        "## Per-Query Comparison",
        "",
        "| Query | Expected | Hybrid Top-1 | Hybrid Exact | Hybrid Struct Score | Cosine Top-1 | Cosine Exact | Cosine Struct Score |",
        "| --- | --- | --- | ---: | ---: | --- | ---: | ---: |",
    ]

    hybrid_rows = {row["query_id"]: row for row in results["per_query"]["hybrid"]}
    cosine_rows = {row["query_id"]: row for row in results["per_query"]["pure_cosine"]}
    for query_id in hybrid_rows:
        hybrid_row = hybrid_rows[query_id]
        cosine_row = cosine_rows[query_id]
        lines.append(
            f"| {query_id} | {hybrid_row['expected_schema_id']} | "
            f"{hybrid_row['predicted_schema_id']} | {hybrid_row['exact_match']} | "
            f"{_format_metric(hybrid_row['metrics']['struct_score'])} | "
            f"{cosine_row['predicted_schema_id']} | {cosine_row['exact_match']} | "
            f"{_format_metric(cosine_row['metrics']['struct_score'])} |"
        )

    return "\n".join(lines) + "\n"


def build_matcher(args: argparse.Namespace) -> HybridSchemaMatcher:
    schema_records = load_schema_records(args.db)
    llm_client = None
    if not args.disable_llm:
        llm_client = LocalOpenAICompatibleLLM(
            api_base=args.api_base,
            model=args.model,
            api_key=args.api_key,
            timeout_seconds=args.timeout,
        )
    return HybridSchemaMatcher(schema_records=schema_records, llm_client=llm_client)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare hybrid schema matching against a pure cosine baseline."
    )
    parser.add_argument("--db", required=True, help="Path to the schema database JSON file.")
    parser.add_argument("--queries", required=True, help="Path to labeled evaluation queries JSON.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of vector candidates to recall before reranking.",
    )
    parser.add_argument(
        "--api-base",
        default="http://127.0.0.1:1234/v1",
        help="OpenAI-compatible API base URL for the hybrid method.",
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
        help="Run the hybrid matcher without the LLM reranker.",
    )
    parser.add_argument(
        "--markdown-out",
        default="",
        help="Optional path to write a Markdown report with paper-ready tables.",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional path to write the raw JSON evaluation output.",
    )

    args = parser.parse_args()
    matcher = build_matcher(args)
    queries = load_queries(args.queries)

    hybrid = evaluate_queries(matcher, queries, method="hybrid", top_k=args.top_k)
    pure_cosine = evaluate_queries(matcher, queries, method="pure_cosine", top_k=args.top_k)

    output = {
        "db": args.db,
        "queries": args.queries,
        "methods": {
            "hybrid": hybrid["aggregate"],
            "pure_cosine": pure_cosine["aggregate"],
        },
        "delta": {
            "sma_at_1": hybrid["aggregate"]["sma_at_1"] - pure_cosine["aggregate"]["sma_at_1"],
            "strict_win_rate_at_1": hybrid["aggregate"]["strict_win_rate_at_1"]
            - pure_cosine["aggregate"]["strict_win_rate_at_1"],
            "top1_exact_accuracy": hybrid["aggregate"]["top1_exact_accuracy"]
            - pure_cosine["aggregate"]["top1_exact_accuracy"],
        },
        "per_query": {
            "hybrid": hybrid["per_query"],
            "pure_cosine": pure_cosine["per_query"],
        },
    }

    if args.json_out:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.markdown_out:
        markdown_path = Path(args.markdown_out)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_markdown_report(output), encoding="utf-8")

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
