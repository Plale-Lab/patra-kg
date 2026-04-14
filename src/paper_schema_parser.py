"""Tabular and document-driven schema extraction for PATRA.

PATRA supports any research domain. Query schemas may be supplied as JSON,
Markdown pipe tables, **comma-separated field identifiers** (pasted schema text),
or DOCX/HTML tables with arbitrary property names.

``CANONICAL_FIELD_GROUPS`` maps a **baseline** set of agriculture/climate-style
synonyms to shared names for demos; **all other columns pass through** as
first-class JSON properties (``confidence: medium``) so HPC, telemetry, and other
domains can run schema search without extending that list.
"""

import json
import re
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from xml.etree import ElementTree

WORD_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


@dataclass
class FieldGroup:
    source_name: str
    canonical_name: str
    json_type: str
    description: str
    series_kind: str = "scalar"
    confidence: str = "high"
    aliases: Tuple[str, ...] = ()

    def to_schema_property(self) -> Dict[str, Any]:
        prop: Dict[str, Any] = {
            "type": self.json_type,
            "description": self.description,
            "x-patra-source-name": self.source_name,
            "x-patra-canonical-name": self.canonical_name,
            "x-patra-confidence": self.confidence,
        }
        if self.aliases:
            prop["x-patra-aliases"] = list(self.aliases)
        if self.series_kind != "scalar":
            prop["x-patra-series-kind"] = self.series_kind
        if self.json_type == "array":
            prop["items"] = {"type": "number"}
        return prop


@dataclass
class ExtractionIssue:
    row_label: str
    reason: str


@dataclass
class SchemaExtractionResult:
    grouped_schema: Dict[str, Any]
    machine_schema: Dict[str, Any]
    grouped_fields: List[FieldGroup]
    provenance: List[Dict[str, Any]]
    unresolved_fields: List[ExtractionIssue]
    confidence: str
    rejected: bool
    rejection_reason: str = ""
    source_kind: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "grouped_schema": self.grouped_schema,
            "machine_schema": self.machine_schema,
            "grouped_fields": [asdict(item) for item in self.grouped_fields],
            "provenance": self.provenance,
            "unresolved_fields": [asdict(item) for item in self.unresolved_fields],
            "confidence": self.confidence,
            "rejected": self.rejected,
            "rejection_reason": self.rejection_reason,
            "source_kind": self.source_kind,
        }


CANONICAL_FIELD_GROUPS = [
    {
        "canonical": "LAT",
        "json_type": "number",
        "series_kind": "scalar",
        "patterns": [r"\blat\b", r"\blatitude\b"],
        "aliases": ("lat", "latitude"),
        "description": "Latitude.",
    },
    {
        "canonical": "LON",
        "json_type": "number",
        "series_kind": "scalar",
        "patterns": [r"\blon\b", r"\blong\b", r"\blongitude\b"],
        "aliases": ("lon", "long", "longitude"),
        "description": "Longitude.",
    },
    {
        "canonical": "DEM",
        "json_type": "number",
        "series_kind": "scalar",
        "patterns": [r"\bdem\b", r"\belevation\b"],
        "aliases": ("dem", "elevation"),
        "description": "Elevation in meters.",
    },
    {
        "canonical": "Tmax_monthly",
        "json_type": "array",
        "series_kind": "monthly_series",
        "patterns": [r"\btmax\b", r"max temperature", r"highest temperature", r"maximum temperature"],
        "aliases": ("tmax", "maximum_temperature"),
        "description": "Maximum temperature per growth month.",
    },
    {
        "canonical": "Tmin_monthly",
        "json_type": "array",
        "series_kind": "monthly_series",
        "patterns": [r"\btmin\b", r"min temperature", r"minimum temperature", r"lowest temperature"],
        "aliases": ("tmin", "minimum_temperature"),
        "description": "Minimum temperature per growth month.",
    },
    {
        "canonical": "SSD_monthly",
        "json_type": "array",
        "series_kind": "monthly_series",
        "patterns": [r"\bssd\b", r"sunshine"],
        "aliases": ("ssd", "sunshine_duration", "sunshine_hours"),
        "description": "Sunshine duration per growth month.",
    },
    {
        "canonical": "PRE_monthly",
        "json_type": "array",
        "series_kind": "monthly_series",
        "patterns": [r"\bpre\b", r"precip"],
        "aliases": ("pre", "precipitation"),
        "description": "Precipitation per growth month.",
    },
    {
        "canonical": "SM_monthly",
        "json_type": "array",
        "series_kind": "monthly_series",
        "patterns": [r"\bsm\b", r"soil moisture"],
        "aliases": ("sm", "soil_moisture"),
        "description": "Soil moisture per growth month.",
    },
    {
        "canonical": "NDVI_monthly",
        "json_type": "array",
        "series_kind": "monthly_series",
        "patterns": [r"\bndvi\b", r"vegetation index"],
        "aliases": ("ndvi", "vegetation_index"),
        "description": "Vegetation index per growth month.",
    },
    {
        "canonical": "Year",
        "json_type": "integer",
        "series_kind": "scalar",
        "patterns": [r"\byear\b", r"growing season year"],
        "aliases": ("year", "season_year"),
        "description": "Growing season year.",
    },
    {
        "canonical": "yield",
        "json_type": "number",
        "series_kind": "scalar",
        "patterns": [r"\byield\b", r"kg/ha"],
        "aliases": ("yield", "yield_kg_ha"),
        "description": "Crop yield in kilograms per hectare.",
    },
]


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def _json_property_name(label: str) -> str:
    """Map an arbitrary column label to a JSON Schema property key (snake_case)."""
    raw = _normalize_text(label)
    if not raw:
        return ""
    s = raw.lower().replace(" ", "_").replace("-", "_")
    s = re.sub(r"[^a-z0-9_]", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        return ""
    if s[0].isdigit():
        s = "f_" + s
    return s


def _field_names_from_plaintext(text: str) -> List[str]:
    """Parse a comma / whitespace / semicolon separated list of identifiers (no pipe tables).

    Used when the file is not a Markdown pipe table but a bare field list such as
    ``job_id,submit_time,wait_time`` from the schema-search text box.
    """
    text = text.strip()
    if not text:
        return []
    if any(line.strip().startswith("|") for line in text.splitlines() if line.strip()):
        return []
    raw_parts = re.split(r"[\s,;]+", text)
    tokens = [p for p in raw_parts if p]
    if not tokens:
        return []
    ident = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    if not all(ident.fullmatch(t) for t in tokens):
        return []
    seen: set[str] = set()
    out: List[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _parse_type(type_text: str) -> str:
    lowered = type_text.lower()
    if "int" in lowered:
        return "integer"
    if "array" in lowered:
        return "array"
    if "float" in lowered or "double" in lowered or "number" in lowered:
        return "number"
    return "string"


def _match_canonical_group(name: str, description: str) -> Optional[Dict[str, Any]]:
    text = f"{name} {description}".lower()
    for spec in CANONICAL_FIELD_GROUPS:
        if any(re.search(pattern, text) for pattern in spec["patterns"]):
            return spec
    return None


def _build_group(
    source_name: str,
    type_text: str,
    description: str,
) -> Optional[FieldGroup]:
    matched = _match_canonical_group(source_name, description)
    if matched:
        json_type = matched["json_type"]
        parsed_type = _parse_type(type_text)
        if json_type == "number" and parsed_type == "integer":
            json_type = "integer"

        return FieldGroup(
            source_name=_normalize_text(source_name),
            canonical_name=matched["canonical"],
            json_type=json_type,
            description=_normalize_text(description) or matched["description"],
            series_kind=matched["series_kind"],
            confidence="high",
            aliases=tuple(matched["aliases"]),
        )

    # Passthrough: HPC, queue logs, and other domains outside CANONICAL_FIELD_GROUPS.
    safe = _json_property_name(source_name)
    if not safe:
        return None
    parsed_type = _parse_type(type_text)
    if parsed_type == "integer":
        json_type = "integer"
    elif parsed_type == "number":
        json_type = "number"
    elif parsed_type == "array":
        json_type = "array"
    else:
        json_type = "string"

    return FieldGroup(
        source_name=_normalize_text(source_name),
        canonical_name=safe,
        json_type=json_type,
        description=_normalize_text(description) or safe,
        series_kind="scalar",
        confidence="medium",
        aliases=(),
    )


def _schema_from_groups(groups: Iterable[FieldGroup], title: str) -> Dict[str, Any]:
    group_list = list(groups)
    return {
        "type": "object",
        "description": title,
        "properties": {
            item.canonical_name: item.to_schema_property()
            for item in group_list
        },
        "required": [item.canonical_name for item in group_list],
        "additionalProperties": False,
    }


def _read_docx_tables(path: Path) -> List[List[List[str]]]:
    with zipfile.ZipFile(path) as archive:
        with archive.open("word/document.xml") as handle:
            root = ElementTree.fromstring(handle.read())

    tables: List[List[List[str]]] = []
    for table in root.findall(".//w:tbl", WORD_NS):
        parsed_rows: List[List[str]] = []
        for row in table.findall("./w:tr", WORD_NS):
            cells: List[str] = []
            for cell in row.findall("./w:tc", WORD_NS):
                text = "".join(node.text or "" for node in cell.findall(".//w:t", WORD_NS))
                cells.append(_normalize_text(text))
            if any(cell for cell in cells):
                parsed_rows.append(cells)
        if parsed_rows:
            tables.append(parsed_rows)
    return tables


def _read_markdown_table(text: str) -> List[List[str]]:
    rows: List[List[str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or stripped.count("|") < 2:
            continue
        if re.fullmatch(r"\|?[\s:-]+\|[\s|:-]*", stripped):
            continue
        cells = [_normalize_text(cell) for cell in stripped.strip("|").split("|")]
        rows.append(cells)
    return rows


def _find_schema_table(tables: List[List[List[str]]]) -> Optional[List[List[str]]]:
    best_table: Optional[List[List[str]]] = None
    best_score = -1
    for table in tables:
        if not table:
            continue
        header = " ".join(cell.lower() for cell in table[0])
        score = 0
        if "column" in header or "feature" in header:
            score += 2
        if "type" in header:
            score += 2
        if "description" in header:
            score += 2
        score += min(len(table), 10)
        if score > best_score:
            best_table = table
            best_score = score
    return best_table


def _first_matching_index(header: List[str], candidates: Tuple[str, ...]) -> Optional[int]:
    for idx, cell in enumerate(header):
        for candidate in candidates:
            if candidate in cell:
                return idx
    return None


def _extract_groups_from_docx_feature_tables(
    tables: List[List[List[str]]],
) -> Tuple[List[FieldGroup], List[ExtractionIssue]]:
    groups: List[FieldGroup] = []
    issues: List[ExtractionIssue] = []
    seen = set()

    for table in tables:
        if len(table) < 2:
            continue
        header = [cell.lower() for cell in table[0]]
        name_index = _first_matching_index(header, ("feature", "variable"))
        abbreviation_index = _first_matching_index(header, ("abbreviation",))
        if name_index is None:
            continue

        type_text = "float"
        if any("encoding" in cell for cell in header):
            type_text = "int"

        for row in table[1:]:
            if not row:
                continue
            row_values = row + [""] * (len(header) - len(row))
            name = row_values[name_index].strip()
            abbreviation = row_values[abbreviation_index].strip() if abbreviation_index is not None else ""
            if not name:
                continue

            row_label = " ".join(part for part in (name, abbreviation) if part).strip()
            if row_label.lower() in {"total", "all aggregated"}:
                continue

            description_parts = []
            for idx, value in enumerate(row_values):
                if idx == name_index:
                    continue
                if value:
                    description_parts.append(value)
            description = " | ".join(description_parts)
            group = _build_group(row_label, type_text, description)
            if group is None:
                issues.append(
                    ExtractionIssue(
                        row_label=_normalize_text(row_label),
                        reason="Could not map this DOCX feature row to a canonical PATRA field group.",
                    )
                )
                continue
            if group.canonical_name in seen:
                continue
            seen.add(group.canonical_name)
            groups.append(group)

    return groups, issues


def _extract_rows_from_table(table: List[List[str]]) -> Tuple[List[FieldGroup], List[ExtractionIssue]]:
    if not table or len(table) < 2:
        return [], [ExtractionIssue(row_label="", reason="No table rows were found.")]

    header = [cell.lower() for cell in table[0]]
    try:
        name_index = next(
            idx for idx, cell in enumerate(header) if cell in {"column", "feature", "field"}
        )
    except StopIteration:
        name_index = 0

    try:
        type_index = next(idx for idx, cell in enumerate(header) if "type" in cell)
    except StopIteration:
        type_index = min(1, len(header) - 1)

    try:
        description_index = next(idx for idx, cell in enumerate(header) if "description" in cell)
    except StopIteration:
        description_index = min(2, len(header) - 1)

    groups: List[FieldGroup] = []
    issues: List[ExtractionIssue] = []
    seen = set()
    for row in table[1:]:
        if not row:
            continue
        name = row[name_index] if name_index < len(row) else row[0]
        type_text = row[type_index] if type_index < len(row) else "string"
        description = row[description_index] if description_index < len(row) else ""
        group = _build_group(name, type_text, description)
        if group is None:
            issues.append(
                ExtractionIssue(
                    row_label=_normalize_text(name),
                    reason="Could not map this row to a canonical PATRA field group.",
                )
            )
            continue
        if group.canonical_name in seen:
            continue
        seen.add(group.canonical_name)
        groups.append(group)

    return groups, issues


def _result_from_groups(
    groups: List[FieldGroup],
    issues: List[ExtractionIssue],
    source_kind: str,
    title: str,
) -> SchemaExtractionResult:
    rejected = not groups
    confidence = "high" if groups and not issues else "medium" if groups else "reject"
    return SchemaExtractionResult(
        grouped_schema=_schema_from_groups(groups, title),
        machine_schema=_schema_from_groups(groups, title),
        grouped_fields=groups,
        provenance=[
            {
                "canonical_name": item.canonical_name,
                "source_name": item.source_name,
                "series_kind": item.series_kind,
                "confidence": item.confidence,
                "method": "deterministic_table_parsing",
            }
            for item in groups
        ],
        unresolved_fields=issues,
        confidence=confidence,
        rejected=rejected,
        rejection_reason="No canonical field groups could be extracted." if rejected else "",
        source_kind=source_kind,
    )


def extract_schema_from_document(path: str) -> SchemaExtractionResult:
    source = Path(path)
    suffix = source.suffix.lower()
    title = f"PATRA query schema extracted from {source.name}"

    if suffix == ".docx":
        tables = _read_docx_tables(source)
        groups, issues = _extract_groups_from_docx_feature_tables(tables)
        if not groups:
            table = _find_schema_table(tables)
            if table is None:
                return SchemaExtractionResult(
                    grouped_schema={},
                    machine_schema={},
                    grouped_fields=[],
                    provenance=[],
                    unresolved_fields=[],
                    confidence="reject",
                    rejected=True,
                    rejection_reason="No candidate schema table found in the DOCX document.",
                    source_kind="docx",
                )
            groups, issues = _extract_rows_from_table(table)
        return _result_from_groups(groups, issues, "docx", title)

    if suffix in {".md", ".txt"}:
        text = source.read_text(encoding="utf-8")
        table = _read_markdown_table(text)
        if not table or len(table) < 2:
            flat_fields = _field_names_from_plaintext(text)
            if flat_fields:
                table = [["column", "type", "description"]] + [
                    [name, "string", name] for name in flat_fields
                ]
        groups, issues = _extract_rows_from_table(table)
        return _result_from_groups(groups, issues, "markdown", title)

    if suffix == ".json":
        data = json.loads(source.read_text(encoding="utf-8"))
        return SchemaExtractionResult(
            grouped_schema=data,
            machine_schema=data,
            grouped_fields=[],
            provenance=[{"method": "direct_json_load", "source_name": source.name}],
            unresolved_fields=[],
            confidence="high",
            rejected=False,
            source_kind="json",
        )

    return SchemaExtractionResult(
        grouped_schema={},
        machine_schema={},
        grouped_fields=[],
        provenance=[],
        unresolved_fields=[],
        confidence="reject",
        rejected=True,
        rejection_reason=f"Unsupported document type: {suffix or 'unknown'}",
        source_kind=suffix.lstrip("."),
    )
