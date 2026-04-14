import argparse
import csv
import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from src.gen_parallel_workloads_benchmark import build_schema_from_csv
    from src.hybrid_schema_matcher import (
        HybridSchemaMatcher,
        LocalOpenAICompatibleLLM,
        MatchResult,
        load_schema,
    )
else:
    from .gen_parallel_workloads_benchmark import build_schema_from_csv
    from .hybrid_schema_matcher import (
        HybridSchemaMatcher,
        LocalOpenAICompatibleLLM,
        MatchResult,
        load_schema,
    )


@dataclass
class DatasetSchemaRecord:
    schema_id: str
    site: str
    split: str
    generator: str
    csv_path: str
    schema: Dict[str, Any]


def _slugify_site(site: str) -> str:
    return site.lower().replace("-", "_").replace(" ", "_")


def _generator_from_path(path: Path) -> str:
    stem = path.stem
    if stem.startswith("synthetic_data_"):
        trimmed = stem[len("synthetic_data_") :]
        return trimmed.split("_")[0]
    return "training"


def discover_dataset_schemas(repo_dir: Path) -> List[DatasetSchemaRecord]:
    records: List[DatasetSchemaRecord] = []
    for site_dir in sorted(path for path in repo_dir.iterdir() if path.is_dir()):
        site = site_dir.name
        site_slug = _slugify_site(site)

        training_dir = site_dir / "training_data"
        if training_dir.exists():
            for csv_path in sorted(training_dir.glob("*.csv")):
                records.append(
                    DatasetSchemaRecord(
                        schema_id=f"{site_slug}_training",
                        site=site,
                        split="training",
                        generator="training",
                        csv_path=str(csv_path),
                        schema=build_schema_from_csv(csv_path),
                    )
                )

        generated_dir = site_dir / "generated_data"
        if generated_dir.exists():
            for csv_path in sorted(generated_dir.glob("*.csv")):
                generator = _generator_from_path(csv_path)
                records.append(
                    DatasetSchemaRecord(
                        schema_id=f"{site_slug}_generated_{generator}",
                        site=site,
                        split="generated",
                        generator=generator,
                        csv_path=str(csv_path),
                        schema=build_schema_from_csv(csv_path),
                    )
                )
    return records


def export_assets(
    dataset_records: List[DatasetSchemaRecord],
    target_schema_path: Path,
    asset_dir: Path,
) -> Tuple[Path, Path]:
    asset_dir.mkdir(parents=True, exist_ok=True)
    db_path = asset_dir / "schema_database.json"
    query_path = asset_dir / "query_schema.json"
    db_path.write_text(
        json.dumps(
            [
                {
                    "id": record.schema_id,
                    "schema": record.schema,
                    "meta": {
                        "site": record.site,
                        "split": record.split,
                        "generator": record.generator,
                        "csv_path": record.csv_path,
                    },
                }
                for record in dataset_records
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    shutil.copyfile(target_schema_path, query_path)
    return db_path, query_path


def build_matcher(
    dataset_records: List[DatasetSchemaRecord],
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
        schema_records=[
            {
                "id": record.schema_id,
                "schema": record.schema,
            }
            for record in dataset_records
        ],
        llm_client=llm_client,
    )


def _schema_meta(dataset_records: List[DatasetSchemaRecord]) -> Dict[str, DatasetSchemaRecord]:
    return {record.schema_id: record for record in dataset_records}


def _short_field_name(path: str) -> str:
    return path.replace("$.", "")


def _characteristics(row: Dict[str, Any]) -> str:
    features: List[str] = []
    aligned_pairs = row.get("aligned_pairs", [])
    if aligned_pairs:
        trimmed = []
        for item in aligned_pairs[:3]:
            pair = item.split(" (", 1)[0]
            left, right = pair.split(" -> ", 1)
            trimmed.append(f"{_short_field_name(left)}->{_short_field_name(right)}")
        features.append("; ".join(trimmed))
    if row.get("derived_support"):
        features.append("derived queue_wait support")
    if row.get("matched_fields"):
        features.append(f"{len(row['matched_fields'])} exact field hits")
    return " | ".join(features) if features else "limited structural support"


def _tradeoffs(row: Dict[str, Any]) -> str:
    notes = list(row.get("tradeoffs", []))
    if row.get("missing_fields"):
        missing = ", ".join(_short_field_name(path) for path in row["missing_fields"][:4])
        notes.append(f"unsupported examples: {missing}")
    return " | ".join(notes[:3]) if notes else "none"


def _escape_cell(text: str) -> str:
    return text.replace("|", "/")


def _rows_from_result(
    result: MatchResult,
    meta_by_id: Dict[str, DatasetSchemaRecord],
    top_k: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in result.report.get("ranking", [])[:top_k]:
        meta = meta_by_id[item["schema_id"]]
        rows.append(
            {
                "rank": item["rank"],
                "schema_id": item["schema_id"],
                "site": meta.site,
                "split": meta.split,
                "generator": meta.generator,
                "csv_path": meta.csv_path,
                "overall_score": item["overall_score"],
                "decision": item["decision"],
                "summary": item["summary"],
                "characteristics": _characteristics(item),
                "tradeoffs": _tradeoffs(item),
                "aligned_pairs": item.get("aligned_pairs", []),
                "derived_support": item.get("derived_support", []),
                "missing_fields": item.get("missing_fields", []),
                "type_conflicts": item.get("type_conflicts", []),
            }
        )
    return rows


def _site_representatives(rows: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    representatives: List[Dict[str, Any]] = []
    seen_sites = set()
    for row in rows:
        if row["site"] in seen_sites:
            continue
        representatives.append(row)
        seen_sites.add(row["site"])
        if len(representatives) >= top_k:
            break
    return representatives


def _generator_display_name(generator: str) -> str:
    mapping = {
        "training": "training",
        "gan": "GAN",
        "ctgan": "CTGAN",
        "tvae": "TVAE",
        "gc": "GC",
        "cg": "CGAN",
        "lublin": "Lublin",
    }
    return mapping.get(generator, generator.upper())


def _schema_display_name(row: Dict[str, Any]) -> str:
    if row["split"] == "training":
        return f"{row['site']} training"
    return f"{row['site']} generated ({_generator_display_name(row['generator'])})"


def _summary_rows(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    hybrid_representatives = [
        row
        for row in results["methods"]["hybrid"]["site_representatives"]
        if row["overall_score"] >= 0.25
    ][:4]
    hybrid_lower_coverage = next(
        (
            row
            for row in results["methods"]["hybrid"]["site_representatives"]
            if row["overall_score"] < 0.25
        ),
        None,
    )
    pure_cosine_top1 = results["methods"]["pure_cosine"]["all_candidates"][0]

    for row in hybrid_representatives:
        rows.append(
            {
                "label": _schema_display_name(row),
                "hybrid_representative": row["overall_score"],
                "hybrid_lower_coverage": "",
                "pure_cosine_top1": "",
            }
        )
    if hybrid_lower_coverage:
        rows.append(
            {
                "label": _schema_display_name(hybrid_lower_coverage),
                "hybrid_representative": "",
                "hybrid_lower_coverage": hybrid_lower_coverage["overall_score"],
                "pure_cosine_top1": "",
            }
        )
    rows.append(
        {
            "label": _schema_display_name(pure_cosine_top1),
            "hybrid_representative": "",
            "hybrid_lower_coverage": "",
            "pure_cosine_top1": pure_cosine_top1["overall_score"],
        }
    )
    return rows


def _latex_score_cell(value: Any) -> str:
    if value == "":
        return ""
    return f"\\textbf{{{float(value):.4f}}}"


def render_summary_latex_table(results: Dict[str, Any]) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Representative schema-search outcomes on the Queue-wait query. Bold scores indicate the selected leading candidate in each method category.}",
        "\\label{tab:queuewait_schema_search_summary}",
        "\\begin{tabular}{lccc}",
        "\\hline",
        " & \\multicolumn{3}{c}{Method} \\\\",
        "\\cline{2-4}",
        "Target Schema & Hybrid representative & Hybrid lower-coverage & Pure cosine top-1 \\\\",
        "\\hline",
    ]
    for row in _summary_rows(results):
        lines.append(
            f"{row['label']} & {_latex_score_cell(row['hybrid_representative'])} & "
            f"{_latex_score_cell(row['hybrid_lower_coverage'])} & "
            f"{_latex_score_cell(row['pure_cosine_top1'])} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def render_full_comparison_markdown(results: Dict[str, Any]) -> str:
    hybrid_rows = {
        row["schema_id"]: row for row in results["methods"]["hybrid"]["all_candidates"]
    }
    cosine_rows = {
        row["schema_id"]: row for row in results["methods"]["pure_cosine"]["all_candidates"]
    }
    ordered_ids = [row["schema_id"] for row in results["methods"]["hybrid"]["all_candidates"]]

    lines = [
        "# Full Queue-Wait Schema Comparison",
        "",
        "| Candidate Schema | Hybrid Score | Hybrid Rank | Hybrid Decision | Pure Cosine Score | Pure Cosine Rank | Pure Cosine Decision |",
        "| --- | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for schema_id in ordered_ids:
        hybrid = hybrid_rows[schema_id]
        cosine = cosine_rows[schema_id]
        lines.append(
            f"| {_schema_display_name(hybrid)} | {hybrid['overall_score']:.4f} | {hybrid['rank']} | "
            f"{hybrid['decision']} | {cosine['overall_score']:.4f} | {cosine['rank']} | "
            f"{cosine['decision']} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_full_comparison_csv(results: Dict[str, Any], path: Path) -> None:
    hybrid_rows = {
        row["schema_id"]: row for row in results["methods"]["hybrid"]["all_candidates"]
    }
    cosine_rows = {
        row["schema_id"]: row for row in results["methods"]["pure_cosine"]["all_candidates"]
    }
    ordered_ids = [row["schema_id"] for row in results["methods"]["hybrid"]["all_candidates"]]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "candidate_schema",
                "site",
                "split",
                "generator",
                "hybrid_score",
                "hybrid_rank",
                "hybrid_decision",
                "pure_cosine_score",
                "pure_cosine_rank",
                "pure_cosine_decision",
            ],
        )
        writer.writeheader()
        for schema_id in ordered_ids:
            hybrid = hybrid_rows[schema_id]
            cosine = cosine_rows[schema_id]
            writer.writerow(
                {
                    "candidate_schema": _schema_display_name(hybrid),
                    "site": hybrid["site"],
                    "split": hybrid["split"],
                    "generator": hybrid["generator"],
                    "hybrid_score": f"{hybrid['overall_score']:.4f}",
                    "hybrid_rank": hybrid["rank"],
                    "hybrid_decision": hybrid["decision"],
                    "pure_cosine_score": f"{cosine['overall_score']:.4f}",
                    "pure_cosine_rank": cosine["rank"],
                    "pure_cosine_decision": cosine["decision"],
                }
            )


def render_markdown_report(results: Dict[str, Any]) -> str:
    lines = [
        "# Queue-Wait Schema Search Report",
        "",
        f"- Query schema: `{results['query_schema']}`",
        f"- Dataset repository: `{results['repo_dir']}`",
        f"- Candidate schemas: `{results['candidate_count']}`",
        f"- LLM enabled: `{results['llm_enabled']}`",
        "",
        "## Hybrid Top-K Results",
        "",
        "| Rank | Dataset | Split | Generator | Score | Decision | Characteristics | Tradeoffs |",
        "| --- | --- | --- | --- | ---: | --- | --- | --- |",
    ]

    for row in results["methods"]["hybrid"]["top_k"]:
        lines.append(
            f"| {row['rank']} | {row['site']} ({row['schema_id']}) | {row['split']} | "
            f"{row['generator']} | {row['overall_score']:.4f} | {row['decision']} | "
            f"{_escape_cell(row['characteristics'])} | {_escape_cell(row['tradeoffs'])} |"
        )

    lines.extend(
        [
            "",
            "## Hybrid Site Representatives",
            "",
            "| Site | Selected Dataset | Split | Generator | Score | Why It Matches |",
            "| --- | --- | --- | --- | ---: | --- |",
        ]
    )
    for row in results["methods"]["hybrid"]["site_representatives"]:
        lines.append(
            f"| {row['site']} | {row['schema_id']} | {row['split']} | {row['generator']} | "
            f"{row['overall_score']:.4f} | {_escape_cell(row['characteristics'])} |"
        )

    lines.extend(
        [
            "",
            "## Pure Cosine Top-K Results",
            "",
            "| Rank | Dataset | Split | Generator | Score | Decision | Characteristics | Tradeoffs |",
            "| --- | --- | --- | --- | ---: | --- | --- | --- |",
        ]
    )
    for row in results["methods"]["pure_cosine"]["top_k"]:
        lines.append(
            f"| {row['rank']} | {row['site']} ({row['schema_id']}) | {row['split']} | "
            f"{row['generator']} | {row['overall_score']:.4f} | {row['decision']} | "
            f"{_escape_cell(row['characteristics'])} | {_escape_cell(row['tradeoffs'])} |"
        )

    lines.extend(
        [
            "",
            "## Comparative Findings",
            "",
            f"- Hybrid rank of first lower-coverage SDSC-95 candidate: `{results['comparative_findings']['hybrid_first_sdsc_rank']}`",
            f"- Non-SDSC candidates placed ahead of SDSC-95 by hybrid: `{results['comparative_findings']['hybrid_non_sdsc_before_sdsc']}`",
            f"- SDSC-95 candidates appearing in pure cosine top-7: `{results['comparative_findings']['pure_cosine_sdsc_in_top_7']}`",
            f"- Best non-SDSC rank under pure cosine: `{results['comparative_findings']['pure_cosine_best_non_sdsc_rank']}`",
            "",
            "## Interpretation",
            "",
            results["interpretation"],
            "",
        ]
    )
    return "\n".join(lines)


def build_interpretation(output: Dict[str, Any]) -> str:
    hybrid_top = output["methods"]["hybrid"]["top_k"][0]
    cosine_top = output["methods"]["pure_cosine"]["top_k"][0]
    findings = output["comparative_findings"]
    return (
        "The hybrid search does not claim an exact schema identity. Instead, it surfaces "
        f"the best partial matches with auditable schema scores. In this run, the top hybrid "
        f"candidate is `{hybrid_top['schema_id']}` with score {hybrid_top['overall_score']:.4f}, "
        "supported by aligned resource-request fields and a derived queue-wait signal from "
        "`wall_time - run_time`. By contrast, the pure cosine baseline ranks "
        f"`{cosine_top['schema_id']}` first with score {cosine_top['overall_score']:.4f}, even "
        "though it covers fewer target concepts. Across the full candidate pool, the hybrid method "
        f"places {findings['hybrid_non_sdsc_before_sdsc']} non-SDSC candidates ahead of the first "
        f"SDSC-95 match, whereas pure cosine places {findings['pure_cosine_sdsc_in_top_7']} SDSC-95 "
        "variants within its top-7 shortlist. This illustrates why the retrieval layer should "
        "return a ranked shortlist rather than a single opaque nearest neighbor."
    )


def build_comparative_findings(output: Dict[str, Any]) -> Dict[str, int]:
    hybrid_rows = output["methods"]["hybrid"]["all_candidates"]
    cosine_rows = output["methods"]["pure_cosine"]["all_candidates"]

    hybrid_first_sdsc_rank = next(
        row["rank"] for row in hybrid_rows if row["site"] == "SDSC-95"
    )
    hybrid_non_sdsc_before_sdsc = sum(
        1 for row in hybrid_rows if row["site"] != "SDSC-95" and row["rank"] < hybrid_first_sdsc_rank
    )
    pure_cosine_sdsc_in_top_7 = sum(
        1 for row in cosine_rows[:7] if row["site"] == "SDSC-95"
    )
    pure_cosine_best_non_sdsc_rank = min(
        row["rank"] for row in cosine_rows if row["site"] != "SDSC-95"
    )
    return {
        "hybrid_first_sdsc_rank": hybrid_first_sdsc_rank,
        "hybrid_non_sdsc_before_sdsc": hybrid_non_sdsc_before_sdsc,
        "pure_cosine_sdsc_in_top_7": pure_cosine_sdsc_in_top_7,
        "pure_cosine_best_non_sdsc_rank": pure_cosine_best_non_sdsc_rank,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search Gen-Parallel-Workloads schemas using the queue-wait query schema."
    )
    parser.add_argument(
        "--repo-dir",
        default="external/Gen-Parallel-Workloads",
        help="Path to the local Gen-Parallel-Workloads checkout.",
    )
    parser.add_argument(
        "--query-schema",
        default="examples/docx_eval/target_eagle_variant.json",
        help="Path to the schema-only query derived from Queue-wait Time Prediction.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of ranked candidates to include in the report tables.",
    )
    parser.add_argument(
        "--asset-dir",
        default="examples/queuewait_schema_search",
        help="Directory where the exported schema database and query schema are written.",
    )
    parser.add_argument(
        "--markdown-out",
        default="reports/queuewait_schema_search_report.md",
        help="Path to write the Markdown report.",
    )
    parser.add_argument(
        "--json-out",
        default="reports/queuewait_schema_search_report.json",
        help="Path to write the JSON report.",
    )
    parser.add_argument(
        "--summary-latex-out",
        default="Patra_REP_2026/tables/queuewait_schema_search_summary_table.tex",
        help="Path to write the paper-ready LaTeX summary table.",
    )
    parser.add_argument(
        "--full-markdown-out",
        default="reports/queuewait_schema_search_full_comparison.md",
        help="Path to write the full all-candidate Markdown comparison table.",
    )
    parser.add_argument(
        "--full-csv-out",
        default="reports/queuewait_schema_search_full_comparison.csv",
        help="Path to write the full all-candidate CSV comparison table.",
    )
    parser.add_argument(
        "--api-base",
        default=os.getenv("LLM_SCHEMA_API_BASE", "http://127.0.0.1:1234/v1"),
        help="OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("LLM_SCHEMA_MODEL", "qwen35-9b"),
        help="Model name exposed by the OpenAI-compatible API.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY", "lm-studio"),
        help="API key for the local OpenAI-compatible endpoint.",
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
        help="Skip LLM reranking and use deterministic hybrid reranking only.",
    )
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir)
    query_schema_path = Path(args.query_schema)
    asset_dir = Path(args.asset_dir)
    markdown_out = Path(args.markdown_out)
    json_out = Path(args.json_out)
    summary_latex_out = Path(args.summary_latex_out)
    full_markdown_out = Path(args.full_markdown_out)
    full_csv_out = Path(args.full_csv_out)

    dataset_records = discover_dataset_schemas(repo_dir)
    meta_by_id = _schema_meta(dataset_records)
    matcher = build_matcher(
        dataset_records=dataset_records,
        api_base=args.api_base,
        model=args.model,
        api_key=args.api_key,
        timeout=args.timeout,
        disable_llm=args.disable_llm,
    )

    export_assets(dataset_records, query_schema_path, asset_dir)
    query_schema = load_schema(str(query_schema_path))
    recall_k = len(dataset_records)
    hybrid_result = matcher.match_schema(query_schema, top_k=recall_k)
    pure_cosine_result = matcher.pure_cosine_match_schema(query_schema, top_k=recall_k)
    hybrid_all_rows = _rows_from_result(hybrid_result, meta_by_id, len(dataset_records))
    pure_cosine_all_rows = _rows_from_result(
        pure_cosine_result, meta_by_id, len(dataset_records)
    )

    output = {
        "repo_dir": str(repo_dir),
        "query_schema": str(query_schema_path),
        "candidate_count": len(dataset_records),
        "llm_enabled": not args.disable_llm,
        "methods": {
            "hybrid": {
                "method": hybrid_result.method,
                "winner": hybrid_result.winner,
                "confidence": hybrid_result.confidence,
                "top_k": hybrid_all_rows[: args.top_k],
                "all_candidates": hybrid_all_rows,
            },
            "pure_cosine": {
                "method": pure_cosine_result.method,
                "winner": pure_cosine_result.winner,
                "confidence": pure_cosine_result.confidence,
                "top_k": pure_cosine_all_rows[: args.top_k],
                "all_candidates": pure_cosine_all_rows,
            },
        },
    }
    output["methods"]["hybrid"]["site_representatives"] = _site_representatives(
        hybrid_all_rows,
        top_k=args.top_k,
    )
    output["methods"]["pure_cosine"]["site_representatives"] = _site_representatives(
        pure_cosine_all_rows,
        top_k=args.top_k,
    )
    output["comparative_findings"] = build_comparative_findings(output)
    output["interpretation"] = build_interpretation(output)
    output["artifacts"] = {
        "summary_latex_table": str(summary_latex_out),
        "full_markdown_comparison": str(full_markdown_out),
        "full_csv_comparison": str(full_csv_out),
    }

    json_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    summary_latex_out.parent.mkdir(parents=True, exist_ok=True)
    full_markdown_out.parent.mkdir(parents=True, exist_ok=True)
    full_csv_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    markdown_out.write_text(render_markdown_report(output), encoding="utf-8")
    summary_latex_out.write_text(render_summary_latex_table(output), encoding="utf-8")
    full_markdown_out.write_text(
        render_full_comparison_markdown(output), encoding="utf-8"
    )
    write_full_comparison_csv(output, full_csv_out)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
