import csv
import json
import os
import urllib.parse
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

if __package__ in {None, ""}:
    from src.gen_parallel_workloads_benchmark import build_schema_from_csv
    from src.paper_schema_parser import FieldGroup
else:
    from .gen_parallel_workloads_benchmark import build_schema_from_csv
    from .paper_schema_parser import FieldGroup


USER_AGENT = "Mozilla/5.0 (PATRA schema pool builder)"


@dataclass
class DatasetSchemaPair:
    dataset_id: str
    title: str
    source_family: str
    source_url: str
    public_access: str
    schema: Dict[str, Any]
    raw_schema: Dict[str, Any]
    task_tags: Dict[str, Any]
    provenance: Dict[str, Any]
    meta: Dict[str, Any]

    def to_manifest_entry(self) -> Dict[str, Any]:
        return asdict(self)

    def to_matcher_record(self) -> Dict[str, Any]:
        return {
            "id": self.dataset_id,
            "schema": self.schema,
            "meta": {
                "title": self.title,
                "source_family": self.source_family,
                "source_url": self.source_url,
                "public_access": self.public_access,
                "task_tags": self.task_tags,
                **self.meta,
            },
        }


def _request(url: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": USER_AGENT})


def _load_json(url: str) -> Dict[str, Any]:
    with urllib.request.urlopen(_request(url), timeout=60) as handle:
        return json.load(handle)


def _download_file(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination
    with urllib.request.urlopen(_request(url), timeout=120) as response:
        destination.write_bytes(response.read())
    return destination


def _canonical_schema(
    title: str,
    description: str,
    fields: Iterable[FieldGroup],
) -> Dict[str, Any]:
    field_list = list(fields)
    return {
        "type": "object",
        "description": f"{title}. {description}",
        "properties": {
            item.canonical_name: item.to_schema_property()
            for item in field_list
        },
        "required": [item.canonical_name for item in field_list],
        "additionalProperties": False,
    }


def _schema_with_fields(
    title: str,
    description: str,
    fields: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    properties: Dict[str, Any] = {}
    required: List[str] = []
    for field in fields:
        name = field["name"]
        json_type = field["type"]
        prop: Dict[str, Any] = {"type": json_type, "description": field["description"]}
        if json_type == "array":
            prop["items"] = {"type": field.get("items_type", "number")}
        properties[name] = prop
        required.append(name)
    return {
        "type": "object",
        "description": f"{title}. {description}",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _field(
    canonical_name: str,
    description: str,
    json_type: str = "number",
    series_kind: str = "scalar",
    aliases: Iterable[str] = (),
) -> FieldGroup:
    return FieldGroup(
        source_name=canonical_name,
        canonical_name=canonical_name,
        json_type=json_type,
        description=description,
        series_kind=series_kind,
        aliases=tuple(aliases),
    )


def _metadata_only_schema(
    title: str,
    description: str,
    fields: Iterable[FieldGroup],
) -> Dict[str, Any]:
    return _canonical_schema(title, description, fields)


def _read_csv_header(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        return next(reader)


def _infer_long_table_schema(path: Path) -> Dict[str, Any]:
    header = _read_csv_header(path)
    properties = {}
    for name in header:
        lowered = name.lower()
        if lowered in {"year", "item code", "area code", "element code"}:
            properties[name] = {"type": "integer", "description": f"Tabular field `{name}`."}
        elif lowered in {"value"}:
            properties[name] = {"type": "number", "description": f"Tabular field `{name}`."}
        else:
            properties[name] = {"type": "string", "description": f"Tabular field `{name}`."}
    return {
        "type": "object",
        "description": "Raw tabular schema inferred from a public FAOSTAT bulk CSV.",
        "properties": properties,
        "required": list(properties.keys()),
        "additionalProperties": False,
    }


def _fetch_text(url: str, max_bytes: int = 4096) -> str:
    with urllib.request.urlopen(_request(url), timeout=60) as handle:
        return handle.read(max_bytes).decode("utf-8", "ignore")


def _build_cybench_pair(cache_dir: Path) -> DatasetSchemaPair:
    record = _load_json("https://zenodo.org/api/records/17279151")
    title = record["metadata"]["title"]
    description = (
        "Public crop-yield forecasting benchmark with subnational wheat and maize yield "
        "targets, growing-season weather indicators, remote sensing indicators, soil "
        "moisture indicators, and static soil properties."
    )
    schema = _metadata_only_schema(
        title,
        description,
        [
            _field("LAT", "Latitude of the administrative unit centroid.", aliases=("latitude",)),
            _field("LON", "Longitude of the administrative unit centroid.", aliases=("longitude",)),
            _field("Year", "Growing season year.", json_type="integer", aliases=("year",)),
            _field(
                "yield",
                "Crop yield target at subnational level.",
                aliases=("yield", "yield_kg_ha"),
            ),
            _field(
                "Tmax_monthly",
                "Growing-season maximum temperature indicators by month.",
                json_type="array",
                series_kind="monthly_series",
                aliases=("tmax",),
            ),
            _field(
                "Tmin_monthly",
                "Growing-season minimum temperature indicators by month.",
                json_type="array",
                series_kind="monthly_series",
                aliases=("tmin",),
            ),
            _field(
                "PRE_monthly",
                "Growing-season precipitation indicators by month.",
                json_type="array",
                series_kind="monthly_series",
                aliases=("precipitation", "pre"),
            ),
            _field(
                "SM_monthly",
                "Growing-season soil moisture indicators by month.",
                json_type="array",
                series_kind="monthly_series",
                aliases=("soil_moisture",),
            ),
            _field(
                "NDVI_monthly",
                "Remote-sensing vegetation indicators by month.",
                json_type="array",
                series_kind="monthly_series",
                aliases=("ndvi", "vegetation_index"),
            ),
        ],
    )
    return DatasetSchemaPair(
        dataset_id="cybench_wheat_subnational",
        title=title,
        source_family="zenodo",
        source_url="https://zenodo.org/records/17279151",
        public_access="public",
        schema=schema,
        raw_schema=schema,
        task_tags={
            "crop": ["wheat", "maize"],
            "task": "subnational crop yield forecasting",
            "temporal_granularity": "growing season monthly indicators",
        },
        provenance={
            "method": "zenodo_record_metadata_inference",
            "record_id": "17279151",
            "notes": "Schema inferred from Zenodo record description because the main archive is large.",
        },
        meta={
            "download_files": [item["key"] for item in record.get("files", [])],
            "cache_dir": str(cache_dir),
        },
    )


def _build_agicam_pair(cache_dir: Path) -> DatasetSchemaPair:
    record = _load_json("https://zenodo.org/api/records/17970104")
    csv_url = "https://zenodo.org/api/records/17970104/files/Winter_wheat_data.csv/content"
    csv_path = _download_file(csv_url, cache_dir / "agicam_winter_wheat.csv")
    raw_schema = build_schema_from_csv(csv_path)
    schema = _schema_with_fields(
        record["metadata"]["title"],
        "Plot-level winter wheat data with vegetation-index metrics, weather measurements, dates, and yield.",
        [
            {
                "name": "observation_date",
                "type": "string",
                "description": "Observation date for time-series aggregation.",
            },
            {
                "name": "vegetation_index_metrics",
                "type": "array",
                "description": "Image-derived vegetation-index measurements across dated observations.",
            },
            {
                "name": "precipitation_observation",
                "type": "number",
                "description": "Observed precipitation at each dated observation.",
            },
            {
                "name": "air_temperature_observation",
                "type": "number",
                "description": "Observed air temperature at each dated observation.",
            },
            {
                "name": "yield",
                "type": "number",
                "description": "Observed wheat yield in kilograms per hectare.",
            },
        ],
    )
    return DatasetSchemaPair(
        dataset_id="agicam_winter_wheat",
        title=record["metadata"]["title"],
        source_family="zenodo",
        source_url="https://zenodo.org/records/17970104",
        public_access="public",
        schema=schema,
        raw_schema=raw_schema,
        task_tags={
            "crop": ["wheat"],
            "task": "yield prediction",
            "spatial_granularity": "plot level",
        },
        provenance={
            "method": "direct_public_csv_schema",
            "record_id": "17970104",
            "file": "Winter_wheat_data.csv",
        },
        meta={
            "local_cache_path": str(csv_path),
            "download_url": csv_url,
            "record_files": [item["key"] for item in record.get("files", [])],
        },
    )


def _build_hf_pair(cache_dir: Path, path: str, dataset_id: str, description: str, fields: Iterable[FieldGroup]) -> DatasetSchemaPair:
    file_url = (
        "https://huggingface.co/datasets/"
        "lafbarbosa/sugarcane_dataset_northeast_sao_paulo_state_brazil/resolve/main/"
        + urllib.parse.quote(path)
    )
    local_name = path.replace("/", "__")
    csv_path = _download_file(file_url, cache_dir / local_name)
    raw_schema = build_schema_from_csv(csv_path)
    title = f"Sugarcane dataset table: {path}"
    return DatasetSchemaPair(
        dataset_id=dataset_id,
        title=title,
        source_family="huggingface",
        source_url="https://huggingface.co/datasets/lafbarbosa/sugarcane_dataset_northeast_sao_paulo_state_brazil",
        public_access="public",
        schema=(
            _canonical_schema(title, description, fields)
            if dataset_id == "sugarcane_production_environment"
            else _schema_with_fields(
                title,
                description,
                [
                    {"name": "LAT", "type": "number", "description": "Latitude."},
                    {"name": "LON", "type": "number", "description": "Longitude."},
                    {
                        "name": "observation_date",
                        "type": "string",
                        "description": "Observation date for daily weather records.",
                    },
                    {
                        "name": "tmax_daily",
                        "type": "number",
                        "description": "Daily maximum temperature observation.",
                    },
                    {
                        "name": "tmin_daily",
                        "type": "number",
                        "description": "Daily minimum temperature observation.",
                    },
                    {
                        "name": "precipitation_daily",
                        "type": "number",
                        "description": "Daily precipitation observation.",
                    },
                ],
            )
        ),
        raw_schema=raw_schema,
        task_tags={
            "crop": ["sugarcane"],
            "task": "yield and production environment modeling",
        },
        provenance={
            "method": "direct_public_csv_schema",
            "file": path,
            "repo": "lafbarbosa/sugarcane_dataset_northeast_sao_paulo_state_brazil",
        },
        meta={
            "local_cache_path": str(csv_path),
            "download_url": file_url,
        },
    )


def _build_sugarcane_pairs(cache_dir: Path) -> List[DatasetSchemaPair]:
    return [
        _build_hf_pair(
            cache_dir=cache_dir,
            path="measurements/meteorological.csv",
            dataset_id="sugarcane_meteorological",
            description=(
                "Daily meteorological observations with coordinates, observation date, "
                "precipitation, Tmax, Tmin, and evapotranspiration."
            ),
            fields=[
                _field("LAT", "Latitude.", aliases=("lat", "latitude")),
                _field("LON", "Longitude.", aliases=("long", "longitude")),
                _field("Year", "Year derivable from observation date and harvest date.", json_type="integer", aliases=("date", "harvest date")),
                _field("Tmax_monthly", "Monthly temperature features derivable from daily Tmax.", json_type="array", series_kind="monthly_series", aliases=("Tmax",)),
                _field("Tmin_monthly", "Monthly temperature features derivable from daily Tmin.", json_type="array", series_kind="monthly_series", aliases=("Tmin",)),
                _field("PRE_monthly", "Monthly precipitation derivable from daily pr.", json_type="array", series_kind="monthly_series", aliases=("pr", "precipitation")),
            ],
        ),
        _build_hf_pair(
            cache_dir=cache_dir,
            path="measurements/production_environment.csv",
            dataset_id="sugarcane_production_environment",
            description=(
                "Production-environment table with coordinates, planting and harvest dates, "
                "soil type, area, and observed yield."
            ),
            fields=[
                _field("LAT", "Latitude.", aliases=("latitude",)),
                _field("LON", "Longitude.", aliases=("longitude",)),
                _field("Year", "Harvest or reference year.", json_type="integer", aliases=("harvest", "harvest_date")),
                _field("yield", "Observed production target.", aliases=("real_TCH", "yield")),
            ],
        ),
    ]


def _build_faostat_pair(cache_dir: Path) -> DatasetSchemaPair:
    zip_url = "https://bulks-faostat.fao.org/production/Production_Crops_Livestock_E_All_Data.zip"
    zip_path = _download_file(zip_url, cache_dir / "faostat_production_crops_livestock.zip")
    with zipfile.ZipFile(zip_path) as archive:
        csv_members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        member = csv_members[0]
        extract_path = cache_dir / member
        if not extract_path.exists():
            archive.extract(member, cache_dir)
    raw_schema = _infer_long_table_schema(extract_path)
    schema = _schema_with_fields(
        "FAOSTAT Production Crops and Livestock",
        "Long-form public agricultural statistics table containing area, item, element, year, unit, and value. It can expose crop yield observations but does not include rich environmental predictors.",
        [
            {"name": "report_year", "type": "integer", "description": "Reported statistical year."},
            {"name": "item_name", "type": "string", "description": "Crop or livestock item name."},
            {"name": "element_name", "type": "string", "description": "Reported indicator or measure such as yield."},
            {"name": "reported_value", "type": "number", "description": "Reported numerical value for the selected item and element."},
        ],
    )
    return DatasetSchemaPair(
        dataset_id="faostat_production_crops_livestock",
        title="FAOSTAT Production Crops and Livestock",
        source_family="fao",
        source_url="https://www.fao.org/faostat/en/#data/QCL",
        public_access="public",
        schema=schema,
        raw_schema=raw_schema,
        task_tags={
            "crop": ["wheat", "general crops"],
            "task": "agricultural production statistics",
            "temporal_granularity": "annual",
        },
        provenance={
            "method": "public_bulk_zip_schema",
            "bulk_download": zip_url,
            "member": member,
        },
        meta={
            "local_cache_path": str(zip_path),
            "extracted_csv_path": str(extract_path),
        },
    )


def _build_lucas_pair() -> DatasetSchemaPair:
    title = "LUCAS 2022 TOPSOIL data"
    schema = _schema_with_fields(
        title,
        "European topsoil survey with geospatial locations and static soil properties. Used as a lower-coverage but public soil-information candidate.",
        [
            {"name": "LAT", "type": "number", "description": "Sampling latitude."},
            {"name": "LON", "type": "number", "description": "Sampling longitude."},
            {"name": "topsoil_properties", "type": "array", "description": "Static topsoil measurements and laboratory soil properties."},
        ],
    )
    return DatasetSchemaPair(
        dataset_id="lucas_2022_topsoil",
        title=title,
        source_family="esdac",
        source_url="https://esdac.jrc.ec.europa.eu/content/lucas-2022-topsoil-data",
        public_access="public-metadata",
        schema=schema,
        raw_schema=schema,
        task_tags={
            "domain": ["soil", "geospatial survey"],
            "task": "soil characterization",
        },
        provenance={
            "method": "public_page_metadata_inference",
            "notes": "Schema inferred from the public dataset overview page because the page exposes overview assets rather than a direct CSV file.",
        },
        meta={},
    )


def build_wheat_vertical_schema_pool(cache_dir: str) -> List[DatasetSchemaPair]:
    """Build the **bundled default public** dataset-schema pool (historical name).

    The pool includes crop, soil, land-use, and related open datasets used to
    exercise schema search across heterogeneous domains. PATRA itself is not
    limited to agriculture; prefer :func:`build_default_public_schema_pool` in new code.
    """
    return build_default_public_schema_pool(cache_dir)


GEN_PARALLEL_WORKLOADS_GITHUB = "https://github.com/DIR-LAB/Gen-Parallel-Workloads"


def build_gen_parallel_workloads_schema_pool(repo_root: str) -> List[DatasetSchemaPair]:
    """Build dataset-schema pairs from a local clone of Gen-Parallel-Workloads.

    Expected layout matches `DIR-LAB/Gen-Parallel-Workloads` (per-site folders with
    ``training_data/*.csv`` and ``generated_data/*.csv``). Schemas are inferred from
    CSV column profiles via :func:`gen_parallel_workloads_benchmark.build_schema_from_csv`.

    Repository: `Gen-Parallel-Workloads <https://github.com/DIR-LAB/Gen-Parallel-Workloads>`_.
    """
    root = Path(repo_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Gen-Parallel-Workloads repo root is not a directory: {root}")
    if __package__ in {None, ""}:
        from src.queuewait_schema_search import discover_dataset_schemas
    else:
        from .queuewait_schema_search import discover_dataset_schemas

    records = discover_dataset_schemas(root)
    pairs: List[DatasetSchemaPair] = []
    for rec in records:
        try:
            rel_csv = str(Path(rec.csv_path).resolve().relative_to(root))
        except ValueError:
            rel_csv = rec.csv_path
        split_label = rec.split
        if rec.split == "generated":
            title = f"Gen-Parallel-Workloads: {rec.site} synthetic trace ({rec.generator})"
        else:
            title = f"Gen-Parallel-Workloads: {rec.site} training trace"
        pairs.append(
            DatasetSchemaPair(
                dataset_id=rec.schema_id,
                title=title,
                source_family="github",
                source_url=GEN_PARALLEL_WORKLOADS_GITHUB,
                public_access="public",
                schema=rec.schema,
                raw_schema=rec.schema,
                task_tags={
                    "domain": ["hpc", "parallel workload", "job trace"],
                    "task": "synthetic job trace generation; scheduling and workload analytics",
                    "site": rec.site,
                    "split": rec.split,
                    "generator": rec.generator,
                },
                provenance={
                    "method": "direct_public_csv_schema",
                    "repository": "DIR-LAB/Gen-Parallel-Workloads",
                    "paper": "Soundar Raj et al., JSSPP 2024 (Empirical Study of ML-based Synthetic Job Trace Generation)",
                },
                meta={
                    "csv_path": rec.csv_path,
                    "repo_relative_csv": rel_csv,
                    "split": split_label,
                },
            )
        )
    return pairs


def build_default_public_schema_pool(
    cache_dir: str,
    gen_parallel_workloads_repo: Optional[str] = None,
) -> List[DatasetSchemaPair]:
    """Return the default bundled public dataset-schema pairs for Agent Tools and demos.

    Deployments may replace this catalog with organization-specific pools; the
    **derivation boundary** (V1) is defined in ``missing_column_derivation``, not by
    this default listing.

    If ``gen_parallel_workloads_repo`` is set to a local path containing a clone of
    `Gen-Parallel-Workloads <https://github.com/DIR-LAB/Gen-Parallel-Workloads>`_,
    those traces are appended to the pool (training and generated splits per site).
    """
    cache = Path(cache_dir)
    out: List[DatasetSchemaPair] = [
        _build_cybench_pair(cache),
        _build_agicam_pair(cache),
        *_build_sugarcane_pairs(cache),
        _build_faostat_pair(cache),
        _build_lucas_pair(),
    ]
    if gen_parallel_workloads_repo:
        root = Path(gen_parallel_workloads_repo)
        if root.is_dir():
            out.extend(build_gen_parallel_workloads_schema_pool(str(root)))
    return out


def write_pool_manifest(pairs: Iterable[DatasetSchemaPair], path: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps([pair.to_manifest_entry() for pair in pairs], indent=2),
        encoding="utf-8",
    )


def build_matcher_records_from_pairs(pairs: Iterable[DatasetSchemaPair]) -> List[Dict[str, Any]]:
    return [pair.to_matcher_record() for pair in pairs]
