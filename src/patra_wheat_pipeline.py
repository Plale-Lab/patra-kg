import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from src.hybrid_schema_matcher import HybridSchemaMatcher, LocalOpenAICompatibleLLM
    from src.missing_column_derivation import analyze_missing_columns, build_derivation_summary
    from src.paper_schema_parser import extract_schema_from_document
    from src.patra_schema_pool import (
        DatasetSchemaPair,
        build_matcher_records_from_pairs,
        build_default_public_schema_pool,
        write_pool_manifest,
    )
else:
    from .hybrid_schema_matcher import HybridSchemaMatcher, LocalOpenAICompatibleLLM
    from .missing_column_derivation import analyze_missing_columns, build_derivation_summary
    from .paper_schema_parser import extract_schema_from_document
    from .patra_schema_pool import (
        DatasetSchemaPair,
        build_matcher_records_from_pairs,
        build_default_public_schema_pool,
        write_pool_manifest,
    )


def _pair_meta(pairs: Iterable[DatasetSchemaPair]) -> Dict[str, DatasetSchemaPair]:
    return {pair.dataset_id: pair for pair in pairs}


def _build_matcher(
    pairs: List[DatasetSchemaPair],
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
    return HybridSchemaMatcher(
        schema_records=build_matcher_records_from_pairs(pairs),
        llm_client=llm_client,
    )


def _candidate_rows(
    result: Dict[str, Any],
    pair_map: Dict[str, DatasetSchemaPair],
    query_schema: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in result["ranking"]:
        pair = pair_map[item["schema_id"]]
        decisions = analyze_missing_columns(query_schema, pair.schema, pair.raw_schema)
        derivation = build_derivation_summary(decisions)
        rows.append(
            {
                "rank": item["rank"],
                "dataset_id": pair.dataset_id,
                "title": pair.title,
                "source_family": pair.source_family,
                "source_url": pair.source_url,
                "score": item["overall_score"],
                "matched_field_groups": [
                    row["target_field"]
                    for row in derivation["rows"]
                    if row["status"] == "directly available"
                ],
                "derivable_field_groups": [
                    row["target_field"]
                    for row in derivation["rows"]
                    if row["status"] == "derivable with provenance"
                ],
                "missing_field_groups": [
                    row["target_field"]
                    for row in derivation["rows"]
                    if row["status"] == "not safely derivable"
                ],
                "derivation_summary": derivation,
                "summary": item["summary"],
                "aligned_pairs": item.get("aligned_pairs", []),
                "derived_support": item.get("derived_support", []),
                "type_conflicts": item.get("type_conflicts", []),
                "tradeoffs": item.get("tradeoffs", []),
            }
        )
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_coverage_matrix(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        header = [
            "dataset_id",
            "title",
            "score",
            "target_field",
            "status",
            "rationale",
            "source_fields",
        ]
        writer.writerow(header)
        for row in rows:
            for decision in row["derivation_summary"]["rows"]:
                writer.writerow(
                    [
                        row["dataset_id"],
                        row["title"],
                        f"{row['score']:.4f}",
                        decision["target_field"],
                        decision["status"],
                        decision["rationale"],
                        "; ".join(decision["source_fields"]),
                    ]
                )


def _write_full_comparison(path: Path, hybrid_rows: List[Dict[str, Any]], cosine_rows: List[Dict[str, Any]]) -> None:
    cosine_by_id = {row["dataset_id"]: row for row in cosine_rows}
    lines = [
        "# PATRA Winter Wheat Full Candidate Comparison",
        "",
        "| Candidate Dataset | Hybrid Score | Hybrid Rank | Direct Groups | Derivable Groups | Missing Groups | Pure Cosine Score | Pure Cosine Rank | Rank Delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in hybrid_rows:
        cosine = cosine_by_id[row["dataset_id"]]
        lines.append(
            f"| {row['title']} | {row['score']:.4f} | {row['rank']} | "
            f"{len(row['matched_field_groups'])} | {len(row['derivable_field_groups'])} | "
            f"{len(row['missing_field_groups'])} | {cosine['score']:.4f} | {cosine['rank']} | "
            f"{cosine['rank'] - row['rank']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_full_comparison_csv(path: Path, hybrid_rows: List[Dict[str, Any]], cosine_rows: List[Dict[str, Any]]) -> None:
    cosine_by_id = {row["dataset_id"]: row for row in cosine_rows}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "dataset_id",
                "title",
                "hybrid_score",
                "hybrid_rank",
                "direct_groups",
                "derivable_groups",
                "missing_groups",
                "pure_cosine_score",
                "pure_cosine_rank",
                "rank_delta",
            ]
        )
        for row in hybrid_rows:
            cosine = cosine_by_id[row["dataset_id"]]
            writer.writerow(
                [
                    row["dataset_id"],
                    row["title"],
                    f"{row['score']:.4f}",
                    row["rank"],
                    len(row["matched_field_groups"]),
                    len(row["derivable_field_groups"]),
                    len(row["missing_field_groups"]),
                    f"{cosine['score']:.4f}",
                    cosine["rank"],
                    cosine["rank"] - row["rank"],
                ]
            )


def _write_shortlist_markdown(path: Path, hybrid_rows: List[Dict[str, Any]], top_k: int) -> None:
    lines = [
        "# PATRA Winter Wheat Shortlist",
        "",
        "| Rank | Dataset | Score | Direct Groups | Derivable Groups | Missing Groups | Source |",
        "| --- | --- | ---: | --- | --- | --- | --- |",
    ]
    for row in hybrid_rows[:top_k]:
        lines.append(
            f"| {row['rank']} | {row['title']} | {row['score']:.4f} | "
            f"{', '.join(row['matched_field_groups']) or 'none'} | "
            f"{', '.join(row['derivable_field_groups']) or 'none'} | "
            f"{', '.join(row['missing_field_groups']) or 'none'} | {row['source_family']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary_table(path: Path, top_rows: List[Dict[str, Any]], cosine_top: Dict[str, Any]) -> None:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{PATRA schema-only retrieval results for the winter-wheat query over the public dataset-schema pool.}",
        "\\label{tab:patra_wheat_schema_search}",
        "\\begin{tabular}{lccc}",
        "\\hline",
        "Candidate Dataset & Hybrid score & Direct/Derivable groups & Pure cosine top-1 baseline \\\\",
        "\\hline",
    ]
    for index, row in enumerate(top_rows):
        baseline = f"\\textbf{{{cosine_top['score']:.4f}}}" if index == 0 else ""
        lines.append(
            f"{row['title']} & \\textbf{{{row['score']:.4f}}} & "
            f"{len(row['matched_field_groups'])}/{len(row['derivable_field_groups'])} & {baseline} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_narrative(path: Path, hybrid_rows: List[Dict[str, Any]], cosine_rows: List[Dict[str, Any]]) -> None:
    top = hybrid_rows[0]
    cosine_top = cosine_rows[0]
    cosine_by_id = {row["dataset_id"]: row for row in cosine_rows}
    deltas = [
        (
            row["dataset_id"],
            row["title"],
            cosine_by_id[row["dataset_id"]]["rank"] - row["rank"],
        )
        for row in hybrid_rows
    ]
    moved_up = max(deltas, key=lambda item: item[2])
    moved_down = min(deltas, key=lambda item: item[2])
    lines = [
        "# PATRA Winter Wheat Narrative",
        "",
        f"- Hybrid top-1 candidate: `{top['dataset_id']}` with score `{top['score']:.4f}`",
        f"- Pure cosine top-1 candidate: `{cosine_top['dataset_id']}` with score `{cosine_top['score']:.4f}`",
        f"- Hybrid direct field groups: `{', '.join(top['matched_field_groups']) or 'none'}`",
        f"- Hybrid derivable field groups: `{', '.join(top['derivable_field_groups']) or 'none'}`",
        f"- Hybrid unresolved field groups: `{', '.join(top['missing_field_groups']) or 'none'}`",
        f"- Largest upward move under hybrid: `{moved_up[1]}` (`{moved_up[2]:+d}` ranks versus pure cosine)",
        f"- Largest downward move under hybrid: `{moved_down[1]}` (`{moved_down[2]:+d}` ranks versus pure cosine)",
        "",
        "The PATRA winter-wheat vertical treats the paper schema as a schema-only query and ranks only public datasets from the curated pool. The highest-ranked candidate is selected because it exposes the broadest support for crop-yield forecasting features while preserving auditable provenance. Relative to pure cosine, the hybrid method rewards candidates with stronger field-group coverage and penalizes candidates that are textually similar but structurally weaker. Missing groups are explicitly separated into derivable versus not safely derivable, so dataset substitution remains traceable rather than implicit.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a PATRA-style public dataset-schema pool and run the winter-wheat schema search vertical."
    )
    parser.add_argument(
        "--query-doc",
        default="examples/patra_wheat/wheat_feature_schema.md",
        help="Path to the paper-derived schema document (.docx, .md, .txt, or .json).",
    )
    parser.add_argument(
        "--cache-dir",
        default="external/patra_public_cache",
        help="Directory used to cache public dataset downloads.",
    )
    parser.add_argument(
        "--gen-parallel-workloads-repo",
        default=os.getenv("GEN_PARALLEL_WORKLOADS_REPO", ""),
        help=(
            "Optional path to a local clone of DIR-LAB/Gen-Parallel-Workloads; "
            "if set, training and generated job-trace CSVs are added to the schema pool. "
            "Overrides env GEN_PARALLEL_WORKLOADS_REPO when non-empty."
        ),
    )
    parser.add_argument(
        "--asset-dir",
        default="examples/patra_wheat",
        help="Directory for generated PATRA wheat assets.",
    )
    parser.add_argument(
        "--report-dir",
        default="reports/patra_wheat",
        help="Directory for generated PATRA wheat reports.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of top candidates to highlight.")
    parser.add_argument(
        "--api-base",
        default=os.getenv("LLM_SCHEMA_API_BASE", "http://127.0.0.1:1234/v1"),
        help="OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("LLM_SCHEMA_MODEL", "qwen35-9b"),
        help="Model name for optional reranking.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY", "lm-studio"),
        help="API key for the local endpoint.",
    )
    parser.add_argument("--timeout", type=int, default=60, help="LLM timeout in seconds.")
    parser.add_argument("--disable-llm", action="store_true", help="Use deterministic reranking only.")
    args = parser.parse_args()

    asset_dir = Path(args.asset_dir)
    report_dir = Path(args.report_dir)

    extraction = extract_schema_from_document(args.query_doc)
    if extraction.rejected:
        raise SystemExit(extraction.rejection_reason)

    gp_repo = (args.gen_parallel_workloads_repo or "").strip() or None
    pairs = build_default_public_schema_pool(args.cache_dir, gen_parallel_workloads_repo=gp_repo)
    pair_map = _pair_meta(pairs)
    matcher = _build_matcher(
        pairs,
        api_base=args.api_base,
        model=args.model,
        api_key=args.api_key,
        timeout=args.timeout,
        disable_llm=args.disable_llm,
    )

    query_schema = extraction.machine_schema
    hybrid = matcher.match_schema(query_schema, top_k=len(pairs))
    cosine = matcher.pure_cosine_match_schema(query_schema, top_k=len(pairs))
    hybrid_rows = _candidate_rows(hybrid.report, pair_map, query_schema)
    cosine_rows = _candidate_rows(cosine.report, pair_map, query_schema)

    asset_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    write_pool_manifest(pairs, str(asset_dir / "schema_pool_manifest.json"))
    _write_json(asset_dir / "query_schema.json", query_schema)
    _write_json(asset_dir / "extraction_report.json", extraction.to_dict())
    _write_json(
        report_dir / "ranking_report.json",
        {
            "query_document": args.query_doc,
            "candidate_count": len(pairs),
            "hybrid": hybrid_rows,
            "pure_cosine": cosine_rows,
        },
    )
    _write_coverage_matrix(report_dir / "coverage_matrix.csv", hybrid_rows)
    _write_shortlist_markdown(report_dir / "shortlist.md", hybrid_rows, args.top_k)
    _write_full_comparison(report_dir / "full_comparison.md", hybrid_rows, cosine_rows)
    _write_full_comparison_csv(report_dir / "full_comparison.csv", hybrid_rows, cosine_rows)
    _write_summary_table(
        Path("Patra_REP_2026/tables/patra_wheat_schema_search.tex"),
        hybrid_rows[: args.top_k],
        cosine_rows[0],
    )
    _write_narrative(report_dir / "narrative.md", hybrid_rows, cosine_rows)
    _write_json(
        report_dir / "derivation_report.json",
        {
            row["dataset_id"]: row["derivation_summary"]
            for row in hybrid_rows[: args.top_k]
        },
    )

    print(
        json.dumps(
            {
                "query_document": args.query_doc,
                "candidate_count": len(pairs),
                "hybrid_top_1": hybrid_rows[0]["dataset_id"],
                "pure_cosine_top_1": cosine_rows[0]["dataset_id"],
                "asset_dir": str(asset_dir),
                "report_dir": str(report_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
