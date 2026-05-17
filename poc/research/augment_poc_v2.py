#!/usr/bin/env python3
"""Data Augmentation POC v2: two-tier HuggingFace → Patra transformation.

Tier 1: Deterministic extraction from HF API JSON (no LLM).
Tier 2: LLM augmentation for fields still null after tier 1.

Output: AssetModelCardCreate-compliant JSON in augmented_cards/.

Usage:
    cd patra-knowledge-base
    python poc/generate_synthetic_dataset.py   # generate dataset first
    python poc/augment_poc_v2.py
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from dotenv import load_dotenv
import httpx

load_dotenv(Path(__file__).parent.parent / ".env")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LITELLM_BASE = "https://litellm.pods.tacc.tapis.io"
TAPIS_AUTH_URL = "https://tacc.tapis.io/v3/oauth2/tokens"
DATASET_PATH = Path(__file__).parent / "data" / "inputs" / "synthetic_hf_cards.json"

# Patra target fields — matches schema.json enums and required fields.
PATRA_TARGET_FIELDS = {
    "name": "Short display name for the model (not the repo path)",
    "version": "Model card version string (e.g. '1.0')",
    "short_description": "One-sentence summary of the model (under 200 chars)",
    "full_description": "2-4 sentence description of what the model does, its architecture, and use cases",
    "keywords": "Comma-separated keywords for discoverability",
    "author": "Creator or organization that published the model",
    "citation": "BibTeX citation entry (@misc or @article) for this model",
    "input_data": "URL or DOI of training dataset (e.g. https://huggingface.co/datasets/imagenet-1k)",
    "input_type": "Primary input data type: Image, Text, Audio, Tabular, Multimodal, or Video",
    "output_data": "Description of model output (e.g. 'class label', 'generated text', 'bounding boxes')",
    "foundational_model": "Base model architecture name if applicable (e.g. ResNet-50, BERT, Llama-3), or null",
    "category": "One of: classification, regression, clustering, anomaly detection, dimensionality reduction, reinforcement learning, natural language processing, computer vision, recommendation systems, time series forecasting, graph learning, graph neural networks, generative modeling, transfer learning, self-supervised learning, semi-supervised learning, unsupervised learning, causal inference, multi-task learning, metric learning, density estimation, multi-label classification, ranking, structured prediction, neural architecture search, sequence modeling, embedding learning, other",
    "documentation": "URL for documentation (typically HuggingFace model page)",
    "is_private": "Whether the model is private (boolean)",
    "is_gated": "Whether the model requires access approval (boolean)",
    "ai_model_framework": "One of: sklearn, tensorflow, pytorch, other",
    "ai_model_license": "License identifier (e.g. apache-2.0, mit, agpl-3.0)",
    "ai_model_model_type": "One of: cnn, decision_tree, dnn, rnn, svm, kmeans, llm, random_forest, lstm, gnn, other",
    "ai_model_version": "Model version string (e.g. '1.0')",
    "ai_model_description": "One-sentence description of the AI model binary",
    "ai_model_owner": "Owner/organization of the model",
    "ai_model_location": "Download URL for the model (typically HuggingFace URL)",
}

MC_REQUIRED_FIELDS = {"name", "category", "input_type", "keywords", "author", "short_description"}

PATRA_DATASHEET_FIELDS = {
    "title": "Dataset title",
    "description": "2-4 sentence description of what the dataset contains and how it can be used",
    "subjects": "Comma-separated topical keywords for discoverability",
    "creator": "Creator name or organization",
    "publisher": "Publishing organization or institution",
    "resource_type": "Specific resource type description (e.g. 'Annotated image dataset for wildlife detection')",
    "resource_type_general": "One of: Dataset, Software, Text, Image, Audiovisual, Collection, Other",
    "publication_year": "Year of publication (integer, e.g. 2025)",
    "size": "Dataset size descriptor (e.g. '250K images, 12GB')",
    "format": "Data format (e.g. CSV, JSON, Parquet, JPEG, WAV, Zarr)",
    "version": "Version string (e.g. '1.0', '2.1')",
    "license": "License identifier (e.g. cc-by-4.0, apache-2.0, cc0-1.0)",
}

DS_REQUIRED_FIELDS = {"title", "description", "subjects", "resource_type_general", "creator"}

DATACITE_TYPES = {"Dataset", "Software", "Text", "Image", "Audiovisual", "Collection", "Other"}
KNOWN_FORMATS = {"csv", "json", "jsonl", "parquet", "hdf5", "zarr", "jpeg", "jpg", "png",
                 "tiff", "geotiff", "wav", "mp3", "flac", "fasta", "pdb", "tsv", "xml"}

# Import mappings from generate_synthetic_dataset
from generate_synthetic_dataset import (
    PIPELINE_TO_CATEGORY, PIPELINE_TO_INPUT_TYPE, PIPELINE_TO_OUTPUT,
    LIBRARY_TO_FRAMEWORK, MODEL_TYPE_MAP, _dataset_to_url,
)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FieldResult:
    field_name: str
    ground_truth: str | None
    augmented_value: str | None
    confidence: float
    heuristic_score: float
    composite_score: float
    exact_match: bool
    semantic_overlap: float
    reasoning: str
    extraction_method: str
    source_score: float = 0.0
    attribute_confidence: float = 0.0

@dataclass
class CardResult:
    card_id: int
    repo_id: str
    asset_type: str
    domain: str
    model_used: str
    tier1_filled: int = 0
    tier2_filled: int = 0
    fields: list[FieldResult] = field(default_factory=list)
    llm_latency_ms: int = 0
    error: str | None = None
    completeness: float = 0.0
    sufficiency: float = 0.0
    overall_confidence: float = 0.0

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def fetch_tapis_token() -> str:
    username = os.getenv("TAPIS_USERNAME", "").strip()
    password = os.getenv("TAPIS_PASSWORD", "").strip()
    if not username or not password:
        print("ERROR: Set TAPIS_USERNAME and TAPIS_PASSWORD in poc/.env")
        sys.exit(1)
    resp = httpx.post(
        TAPIS_AUTH_URL,
        json={"username": username, "password": password, "grant_type": "password"},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["result"]["access_token"]["access_token"]

# ---------------------------------------------------------------------------
# Tier 1: Deterministic extraction from HF API JSON
# ---------------------------------------------------------------------------

def _strip_org(repo_path: str | None) -> str | None:
    if not repo_path:
        return None
    return repo_path.split("/")[-1] if "/" in repo_path else repo_path


def _coerce_is_gated(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() not in {"", "false", "0", "none", "null"}


# ---------------------------------------------------------------------------
# Tag prefix parsing
# ---------------------------------------------------------------------------

def parse_tag_prefixes(tags: list[str]) -> dict:
    result = {"arxiv_ids": [], "base_models": [], "datasets": [], "license": None, "languages": []}
    for tag in tags:
        if tag.startswith("arxiv:"):
            result["arxiv_ids"].append(tag.removeprefix("arxiv:"))
        elif tag.startswith("base_model:"):
            raw = tag.removeprefix("base_model:")
            if raw.startswith("finetune:"):
                raw = raw.removeprefix("finetune:")
            result["base_models"].append(raw)
        elif tag.startswith("dataset:"):
            result["datasets"].append(tag.removeprefix("dataset:"))
        elif tag.startswith("license:"):
            result["license"] = tag.removeprefix("license:")
        elif len(tag) <= 3 and tag.isalpha():
            result["languages"].append(tag)
    return result


# ---------------------------------------------------------------------------
# README frontmatter parsing
# ---------------------------------------------------------------------------

def parse_readme_frontmatter(readme: str) -> dict:
    result = {}
    if not readme or not readme.strip().startswith("---"):
        return result
    lines = readme.strip().split("\n")
    if len(lines) < 3:
        return result
    end_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break
    if end_idx is None:
        return result
    for line in lines[1:end_idx]:
        if ":" in line:
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()
            if key == "base_model" and val and not val.startswith("["):
                if val.startswith("- "):
                    val = val.removeprefix("- ").strip()
                if val:
                    result["base_model"] = val
            elif key == "license" and val:
                result["license"] = val
            elif key == "license_link" and val:
                result["license_link"] = val
            elif key == "pipeline_tag" and val:
                result["pipeline_tag"] = val
    return result


# ---------------------------------------------------------------------------
# README section parsing
# ---------------------------------------------------------------------------

_DESCRIPTION_HEADINGS = {
    "model description", "about", "model overview", "overview", "description",
    "introduction", "model summary", "summary", "what is this",
}
_TRAINING_HEADINGS = {
    "training", "training details", "training data", "training procedure",
    "data", "dataset", "datasets",
}

def parse_readme_sections(readme: str) -> dict[str, str]:
    if not readme:
        return {}
    body = readme.strip()
    if body.startswith("---"):
        end = body.find("---", 3)
        if end != -1:
            body = body[end + 3:].strip()

    sections: dict[str, str] = {}
    current_heading = None
    current_lines: list[str] = []

    for line in body.split("\n"):
        if line.startswith("## "):
            if current_heading is not None:
                sections[current_heading] = "\n".join(current_lines).strip()
            current_heading = line.lstrip("#").strip().lower()
            current_lines = []
        elif current_heading is not None:
            current_lines.append(line)

    if current_heading is not None:
        sections[current_heading] = "\n".join(current_lines).strip()

    return sections


def extract_description_from_sections(sections: dict[str, str]) -> tuple[str | None, str | None]:
    for heading in _DESCRIPTION_HEADINGS:
        if heading in sections:
            text = sections[heading]
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip() and not p.strip().startswith("```") and len(p.strip()) > 20]
            if not paragraphs:
                continue
            short = paragraphs[0]
            if len(short) > 200:
                cut = short[:200].rfind(" ")
                short = short[:cut] + "..." if cut > 50 else short[:200]
            full = " ".join(paragraphs[:3])
            if len(full) > 800:
                full = full[:800].rsplit(" ", 1)[0] + "..."
            return short, full
    return None, None


def extract_first_prose(readme: str) -> tuple[str | None, str | None]:
    if not readme:
        return None, None
    body = readme.strip()
    if body.startswith("---"):
        end = body.find("---", 3)
        if end != -1:
            body = body[end + 3:].strip()

    paragraphs = []
    for line_group in body.split("\n\n"):
        text = line_group.strip()
        if not text or text.startswith("#") or text.startswith("```") or text.startswith("|") or text.startswith("- ") or text.startswith("<"):
            continue
        if len(text) > 30:
            paragraphs.append(text)
        if len(paragraphs) >= 3:
            break

    if not paragraphs:
        return None, None

    short = paragraphs[0]
    if len(short) > 200:
        cut = short[:200].rfind(" ")
        short = short[:cut] + "..." if cut > 50 else short[:200]
    full = " ".join(paragraphs[:3])
    if len(full) > 800:
        full = full[:800].rsplit(" ", 1)[0] + "..."
    return short, full


# ---------------------------------------------------------------------------
# Method 1: Structured Extraction (minimal LLM)
# ---------------------------------------------------------------------------

def structured_extraction(hf_card: dict, readme_body: str) -> dict:
    extracted = {k: None for k in PATRA_TARGET_FIELDS}
    tags = hf_card.get("tags") or []
    repo_id = hf_card.get("id") or hf_card.get("modelId") or ""
    pipeline = hf_card.get("pipeline_tag")

    # Tag prefix parsing
    tag_data = parse_tag_prefixes(tags)

    # Basic deterministic fields
    extracted["name"] = _strip_org(repo_id)
    extracted["author"] = hf_card.get("author") or (repo_id.split("/")[0] if "/" in repo_id else None)
    extracted["documentation"] = f"https://huggingface.co/{repo_id}" if repo_id else None
    extracted["ai_model_location"] = extracted["documentation"]
    extracted["ai_model_owner"] = extracted["author"]
    extracted["ai_model_version"] = "1.0"
    extracted["version"] = "1.0"
    extracted["is_private"] = bool(hf_card.get("private"))
    extracted["is_gated"] = _coerce_is_gated(hf_card.get("gated"))

    # From tag prefixes
    if tag_data["license"]:
        extracted["ai_model_license"] = tag_data["license"]
    if tag_data["arxiv_ids"]:
        extracted["citation"] = f"https://arxiv.org/abs/{tag_data['arxiv_ids'][0]}"
    if tag_data["base_models"]:
        extracted["foundational_model"] = _strip_org(tag_data["base_models"][0])
    if tag_data["datasets"]:
        extracted["input_data"] = ", ".join(f"https://huggingface.co/datasets/{d}" for d in tag_data["datasets"])

    # Pipeline tag lookups (imported from generate_synthetic_dataset)
    if pipeline:
        extracted["category"] = PIPELINE_TO_CATEGORY.get(pipeline)
        extracted["input_type"] = PIPELINE_TO_INPUT_TYPE.get(pipeline)
        extracted["output_data"] = PIPELINE_TO_OUTPUT.get(pipeline)

    # Library → framework
    lib_name = hf_card.get("library_name") or ""
    if lib_name:
        extracted["ai_model_framework"] = LIBRARY_TO_FRAMEWORK.get(lib_name, "other")

    # Keywords from non-prefixed tags
    prefix_set = {"arxiv:", "base_model:", "dataset:", "license:", "region:", "deploy:", "base_model:finetune:"}
    kw_tags = [t for t in tags if not any(t.startswith(p) for p in prefix_set) and t not in ("endpoints_compatible", "text-generation-inference", "safetensors")]
    if kw_tags:
        extracted["keywords"] = ", ".join(kw_tags[:15])

    # README frontmatter
    fm = parse_readme_frontmatter(readme_body)
    if not extracted["foundational_model"] and fm.get("base_model"):
        extracted["foundational_model"] = _strip_org(fm["base_model"])
    if not extracted["ai_model_license"] and fm.get("license"):
        extracted["ai_model_license"] = fm["license"]
    if not extracted["category"] and fm.get("pipeline_tag"):
        extracted["category"] = PIPELINE_TO_CATEGORY.get(fm["pipeline_tag"])
        extracted["input_type"] = PIPELINE_TO_INPUT_TYPE.get(fm["pipeline_tag"])

    # README BibTeX extraction
    extracted = extract_from_readme(readme_body, extracted)

    # README section-based description extraction
    sections = parse_readme_sections(readme_body)
    short, full = extract_description_from_sections(sections)
    if short:
        extracted["short_description"] = short
        extracted["ai_model_description"] = short
    if full:
        extracted["full_description"] = full

    # Fallback: first prose paragraphs if section parsing found nothing
    if not extracted["short_description"]:
        short_fb, full_fb = extract_first_prose(readme_body)
        if short_fb:
            extracted["short_description"] = short_fb
            extracted["ai_model_description"] = short_fb
        if full_fb and not extracted["full_description"]:
            extracted["full_description"] = full_fb

    # Model type from config (if available) or model name heuristics
    config = hf_card.get("config") or {}
    raw_type = config.get("model_type") or ""
    if raw_type:
        extracted["ai_model_model_type"] = MODEL_TYPE_MAP.get(raw_type.lower(), "other")
    elif extracted.get("name"):
        name_lower = extracted["name"].lower()
        for pattern, mtype in MODEL_TYPE_MAP.items():
            if pattern in name_lower:
                extracted["ai_model_model_type"] = mtype
                break

    # ai_model.name
    extracted["ai_model_name"] = extracted.get("name")

    return extracted


def extract_from_hf_json(hf_card: dict) -> dict:
    card_data = hf_card.get("cardData") or {}
    tags = hf_card.get("tags") or []
    config = hf_card.get("config") or {}

    datasets_from_tags = [t.split(":", 1)[1] for t in tags if t.startswith("dataset:")]
    license_from_tags = next((t.split(":", 1)[1] for t in tags if t.startswith("license:")), None)

    keyword_tags = [t for t in (card_data.get("tags") or []) if ":" not in t]
    if not keyword_tags:
        keyword_tags = [t for t in tags
                        if not any(t.startswith(p) for p in ("dataset:", "license:", "arxiv:", "region:"))]

    repo_id = hf_card.get("id") or ""
    name = repo_id.split("/")[-1] if "/" in repo_id else repo_id
    pipeline = hf_card.get("pipeline_tag") or card_data.get("pipeline_tag")

    datasets = card_data.get("datasets") or datasets_from_tags
    dataset_urls = ", ".join(_dataset_to_url(d) for d in datasets) if datasets else None

    lib_name = hf_card.get("library_name") or card_data.get("library_name")
    framework = LIBRARY_TO_FRAMEWORK.get(lib_name, "other") if lib_name else None

    raw_model_type = config.get("model_type")
    if not raw_model_type:
        architectures = config.get("architectures") or []
        raw_model_type = architectures[0].lower() if architectures else None
    model_type = MODEL_TYPE_MAP.get(raw_model_type, "other") if raw_model_type else None

    arxiv_ids = [t.split(":", 1)[1] for t in tags if t.startswith("arxiv:")]
    citation = f"https://arxiv.org/abs/{arxiv_ids[0]}" if arxiv_ids else None

    version = str(card_data.get("version", "")) or "1.0"

    return {
        "name": name or None,
        "version": version,
        "short_description": None,
        "full_description": None,
        "keywords": ", ".join(keyword_tags) if keyword_tags else None,
        "author": hf_card.get("author") or (repo_id.split("/")[0] if "/" in repo_id else None),
        "citation": citation,
        "input_data": dataset_urls,
        "input_type": PIPELINE_TO_INPUT_TYPE.get(pipeline, None) if pipeline else None,
        "output_data": PIPELINE_TO_OUTPUT.get(pipeline, None) if pipeline else None,
        "foundational_model": _strip_org(card_data.get("base_model")),
        "category": PIPELINE_TO_CATEGORY.get(pipeline, None) if pipeline else None,
        "documentation": f"https://huggingface.co/{repo_id}" if repo_id else None,
        "is_private": hf_card.get("private", False),
        "is_gated": _coerce_is_gated(hf_card.get("gated")),
        "ai_model_framework": framework,
        "ai_model_license": card_data.get("license") or license_from_tags or None,
        "ai_model_model_type": model_type,
        "ai_model_version": version,
        "ai_model_description": None,
        "ai_model_owner": hf_card.get("author") or (repo_id.split("/")[0] if "/" in repo_id else None),
        "ai_model_location": f"https://huggingface.co/{repo_id}" if repo_id else None,
        "ai_model_test_accuracy": None,
    }

# ---------------------------------------------------------------------------
# Phase 1.5: README extraction
# ---------------------------------------------------------------------------

def extract_from_readme(readme_body: str | None, extracted: dict) -> dict:
    if not readme_body or not readme_body.strip():
        return extracted

    lines = readme_body.strip().split("\n")

    # citation: extract BibTeX from ```bibtex block
    if extracted.get("citation") is None:
        in_bibtex = False
        bibtex_lines = []
        for line in lines:
            if "```bibtex" in line:
                in_bibtex = True
                continue
            if in_bibtex and line.strip() == "```":
                break
            if in_bibtex:
                bibtex_lines.append(line)
        if bibtex_lines:
            extracted["citation"] = "\n".join(bibtex_lines).strip()

    return extracted


def extract_descriptions_from_readme(readme_body: str | None, extracted: dict) -> dict:
    """B1: Deterministic description extraction from README paragraphs."""
    if not readme_body or not readme_body.strip():
        return extracted

    # Strip YAML frontmatter if present
    body = readme_body.strip()
    if body.startswith("---"):
        end = body.find("---", 3)
        if end != -1:
            body = body[end + 3:].strip()

    # Split into paragraphs (non-empty line groups separated by blank lines)
    paragraphs = []
    current = []
    for line in body.split("\n"):
        stripped = line.strip()
        # Skip headings, code blocks, and list items for paragraph extraction
        if stripped.startswith("#") or stripped.startswith("```") or stripped.startswith("- ") or stripped.startswith("| "):
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        if not stripped:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        current.append(stripped)
    if current:
        paragraphs.append(" ".join(current))

    # Filter to paragraphs with actual prose (>30 chars, not code/imports)
    prose = [p for p in paragraphs if len(p) > 30 and not p.startswith("from ") and not p.startswith("import ")]

    if not prose:
        return extracted

    # short_description: first prose paragraph, truncated to 200 chars
    if extracted.get("short_description") is None:
        first = prose[0]
        if len(first) > 200:
            # Cut at last space before 200
            cut = first[:200].rfind(" ")
            first = first[:cut] + "..." if cut > 50 else first[:200]
        extracted["short_description"] = first

    # full_description: first 2-3 prose paragraphs joined
    if extracted.get("full_description") is None:
        full = " ".join(prose[:3])
        if len(full) > 800:
            full = full[:800].rsplit(" ", 1)[0] + "..."
        extracted["full_description"] = full

    # ai_model_description: same as short_description
    if extracted.get("ai_model_description") is None and extracted.get("short_description"):
        extracted["ai_model_description"] = extracted["short_description"]

    # For datasheets: description from first prose paragraphs
    if extracted.get("description") is None:
        extracted["description"] = " ".join(prose[:2])

    return extracted
# Tier 2: LLM augmentation (only for null fields)
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """You are a metadata specialist for the Patra ML model catalog.

Given this HuggingFace model API response and README content, fill in ONLY the missing fields listed below. Do NOT regenerate fields that are already extracted.

## HuggingFace API response
{hf_json}

## README.md content
{readme_body}

## Fields already extracted (for context only, do NOT change these)
{extracted_json}

## Fields to fill (these are null, generate values for them)
{missing_fields_json}

## Schema notes
- "short_description" is a single sentence (under 200 chars), NOT the same as "full_description"
- "full_description" is 2-4 sentences summarizing the model, its architecture, and use cases
- "ai_model_description" is a one-sentence description of the model binary (can match short_description)
- "ai_model_framework", "ai_model_license", "ai_model_model_type" describe the model binary
- "category" must be one of the allowed enum values
- "input_type" must be one of: Image, Text, Audio, Tabular, Multimodal, Video

## Rules
- Use information from both the HF API response and the README content
- For "short_description", write one concise sentence. Do not copy the full_description.
- For "full_description", write 2-4 sentences. Do not duplicate the short_description verbatim.
- If you truly cannot infer a value, set it to null with confidence 0.0
- "keywords" should be comma-separated, NOT a JSON array
- "foundational_model" should be the architecture name only (e.g. "ResNet-50"), not the full repo path

## Output format
Return ONLY valid JSON, no markdown fences:
{{
  "augmented_fields": {{
    "<field_name>": {{
      "value": "<your value or null>",
      "confidence": 0.85,
      "reasoning": "<one sentence>"
    }}
  }}
}}"""

FEWSHOT_PROMPT_TEMPLATE = """You are a metadata specialist for the Patra ML model catalog.

Here are 3 examples of completed Patra model cards to learn the format and style:

{exemplars}

Now augment this new model card. Fill in ONLY the missing fields listed below, matching the style of the examples above.

## HuggingFace API response
{hf_json}

## README.md content
{readme_body}

## Fields already extracted (do NOT change these)
{extracted_json}

## Fields to fill (generate values matching the exemplar style)
{missing_fields_json}

## Schema constraints
- "category" must be one of: {category_enum}
- "ai_model_framework" must be one of: sklearn, tensorflow, pytorch, other
- "ai_model_model_type" must be one of: cnn, decision_tree, dnn, rnn, svm, kmeans, llm, random_forest, lstm, gnn, other

Return ONLY valid JSON:
{{
  "augmented_fields": {{
    "<field_name>": {{
      "value": "<your value or null>",
      "confidence": 0.85,
      "reasoning": "<one sentence>"
    }}
  }}
}}"""

COT_ANALYSIS_PROMPT = """You are analyzing a HuggingFace model to prepare metadata for the Patra ML catalog.

## HuggingFace API response
{hf_json}

## README.md content
{readme_body}

Analyze this model and answer these questions:
1. What type of model is this? (architecture family, task type, domain)
2. What data was it trained on?
3. What are its key capabilities?
4. What are its limitations?
5. Who created it and why?
6. Is it a fine-tune of another model? If so, which base model?

Write a structured analysis. Do NOT fill any catalog fields yet — just reason about what you know."""

COT_GENERATE_PROMPT = """Based on your analysis below, fill in these Patra catalog fields.

## Your analysis
{analysis}

## Fields already extracted (do NOT change these)
{extracted_json}

## Fields to fill
{missing_fields_json}

## Schema constraints
- "category" must be one of: {category_enum}
- "ai_model_framework" must be one of: sklearn, tensorflow, pytorch, other
- "ai_model_model_type" must be one of: cnn, decision_tree, dnn, rnn, svm, kmeans, llm, random_forest, lstm, gnn, other

For each field, explain your reasoning. If you cannot infer a value, set it to null with confidence 0.0.

Return ONLY valid JSON:
{{
  "augmented_fields": {{
    "<field_name>": {{
      "value": "<your value or null>",
      "confidence": 0.85,
      "reasoning": "<one sentence>"
    }}
  }}
}}"""

COT_VERIFY_PROMPT = """Review these augmented Patra catalog fields for internal consistency.

## All fields (extracted + generated)
{all_fields_json}

Check:
1. Does category match input_type? (e.g., "classification" + "Image" = OK, "classification" + "Audio" = check)
2. Does ai_model_model_type match foundational_model? (e.g., "llm" + "Llama" = OK, "cnn" + "BERT" = wrong)
3. Does the description accurately reflect the category and input_type?
4. Are there any contradictions between fields?

Return ONLY valid JSON:
{{
  "corrections": [
    {{"field": "<field_name>", "old_value": "<current>", "new_value": "<corrected>", "reason": "<why>"}}
  ],
  "consistent": true/false
}}"""


# ---------------------------------------------------------------------------
# Phase 1: Datasheet extraction from HF Dataset API JSON
# ---------------------------------------------------------------------------

def extract_from_hf_dataset_json(hf_ds: dict) -> dict:
    card_data = hf_ds.get("cardData") or {}
    repo_id = hf_ds.get("id", "")
    extracted = {k: None for k in PATRA_DATASHEET_FIELDS}

    extracted["title"] = card_data.get("pretty_name") or _strip_org(repo_id)
    extracted["creator"] = hf_ds.get("author") or (repo_id.split("/")[0] if "/" in repo_id else None)
    extracted["license"] = card_data.get("license")

    task_cats = card_data.get("task_categories") or []
    if task_cats:
        extracted["subjects"] = ", ".join(task_cats)

    size_cats = card_data.get("size_categories")
    if isinstance(size_cats, list) and size_cats:
        extracted["size"] = size_cats[0]
    elif isinstance(size_cats, str):
        extracted["size"] = size_cats

    created = hf_ds.get("createdAt") or ""
    if created and len(created) >= 4:
        try:
            extracted["publication_year"] = str(int(created[:4]))
        except ValueError:
            pass

    return extracted


# Fields that are direct copies from API (confidence = 1.0 when filled)
_DIRECT_COPY_FIELDS = {"name", "author", "documentation", "is_private", "is_gated",
                       "ai_model_owner", "ai_model_location", "ai_model_version", "version"}

# Fields filled via lookup tables (confidence = 0.95 when matched, 0.6 when fallback to "other")
_LOOKUP_FIELDS = {"category", "input_type", "output_data", "ai_model_framework", "ai_model_model_type"}

# Fields extracted from README (confidence = 0.9)
_README_FIELDS = {"short_description", "full_description", "ai_model_description", "citation"}


def compute_attribute_confidence(fname: str, method: str, value, llm_conf: float) -> float:
    if value is None or (isinstance(value, str) and not value.strip()):
        return 0.0
    if method in ("llm", "llm_with_readme"):
        return round(llm_conf, 3)
    if method == "readme_bibtex":
        return 0.9
    # Deterministic extraction — confidence depends on field type
    if fname in _DIRECT_COPY_FIELDS:
        return 1.0
    if fname in _LOOKUP_FIELDS:
        s = str(value).strip().lower()
        return 0.6 if s == "other" else 0.95
    if fname in _README_FIELDS:
        return 0.9
    return 0.9


# ---------------------------------------------------------------------------
# Phase 2: LLM augmentation
# ---------------------------------------------------------------------------

def call_llm(token: str, model: str, hf_card: dict, extracted: dict, missing_fields: dict, readme_body: str = "") -> tuple[dict, int]:
    prompt = PROMPT_TEMPLATE.format(
        hf_json=json.dumps(hf_card, indent=2, default=str),
        readme_body=readme_body[:3000] if readme_body else "(no README available)",
        extracted_json=json.dumps({k: v for k, v in extracted.items() if v is not None}, indent=2),
        missing_fields_json=json.dumps(missing_fields, indent=2),
    )
    t0 = time.time()
    resp = httpx.post(
        f"{LITELLM_BASE}/v1/chat/completions",
        headers={"Content-Type": "application/json", "X-Tapis-Token": token},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 1500,
        },
        timeout=90,
    )
    latency_ms = int((time.time() - t0) * 1000)
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"].strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw), latency_ms


def _raw_llm_call(token: str, model: str, prompt: str, max_tokens: int = 1500, temperature: float = 0.2) -> tuple[str, int]:
    t0 = time.time()
    resp = httpx.post(
        f"{LITELLM_BASE}/v1/chat/completions",
        headers={"Content-Type": "application/json", "X-Tapis-Token": token},
        json={"model": model, "messages": [{"role": "user", "content": prompt}],
              "temperature": temperature, "max_tokens": max_tokens},
        timeout=90,
    )
    latency_ms = int((time.time() - t0) * 1000)
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"].strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return raw, latency_ms


def _category_enum_str():
    from generate_synthetic_dataset import PIPELINE_TO_CATEGORY
    return ", ".join(sorted(set(PIPELINE_TO_CATEGORY.values())))


# Exemplar cache for few-shot (built once from M1 augmented cards)
_fewshot_exemplars: list[dict] | None = None

def _load_fewshot_exemplars() -> list[dict]:
    global _fewshot_exemplars
    if _fewshot_exemplars is not None:
        return _fewshot_exemplars
    augmented_dir = Path(__file__).parent / "data" / "outputs" / "augmented_cards"
    exemplars = []
    for f in sorted(augmented_dir.glob("*.json"))[:5]:
        try:
            exemplars.append(json.loads(f.read_text()))
        except Exception:
            continue
    _fewshot_exemplars = exemplars
    return exemplars


def call_llm_fewshot(token: str, model: str, hf_card: dict, extracted: dict, missing_fields: dict, readme_body: str = "") -> tuple[dict, int]:
    exemplars = _load_fewshot_exemplars()
    exemplar_text = "\n\n".join(f"### Example {i+1}\n```json\n{json.dumps(ex, indent=2)}\n```" for i, ex in enumerate(exemplars[:3]))
    if not exemplar_text:
        exemplar_text = "(no exemplars available — generate fields based on schema descriptions)"
    prompt = FEWSHOT_PROMPT_TEMPLATE.format(
        exemplars=exemplar_text,
        hf_json=json.dumps(hf_card, indent=2, default=str),
        readme_body=readme_body[:3000] if readme_body else "(no README available)",
        extracted_json=json.dumps({k: v for k, v in extracted.items() if v is not None}, indent=2),
        missing_fields_json=json.dumps(missing_fields, indent=2),
        category_enum=_category_enum_str(),
    )
    raw, latency = _raw_llm_call(token, model, prompt)
    return json.loads(raw), latency


def call_llm_cot(token: str, model: str, hf_card: dict, extracted: dict, missing_fields: dict, readme_body: str = "") -> tuple[dict, int]:
    total_latency = 0

    # Pass 1: Analysis
    analysis_prompt = COT_ANALYSIS_PROMPT.format(
        hf_json=json.dumps(hf_card, indent=2, default=str),
        readme_body=readme_body[:3000] if readme_body else "(no README available)",
    )
    analysis, lat1 = _raw_llm_call(token, model, analysis_prompt, max_tokens=800)
    total_latency += lat1

    # Pass 2: Generation
    gen_prompt = COT_GENERATE_PROMPT.format(
        analysis=analysis,
        extracted_json=json.dumps({k: v for k, v in extracted.items() if v is not None}, indent=2),
        missing_fields_json=json.dumps(missing_fields, indent=2),
        category_enum=_category_enum_str(),
    )
    gen_raw, lat2 = _raw_llm_call(token, model, gen_prompt)
    total_latency += lat2
    gen_result = json.loads(gen_raw)

    # Pass 3: Verification
    all_fields = {k: v for k, v in extracted.items() if v is not None}
    for fname, suggestion in gen_result.get("augmented_fields", {}).items():
        if suggestion.get("value") is not None:
            all_fields[fname] = suggestion["value"]

    verify_prompt = COT_VERIFY_PROMPT.format(all_fields_json=json.dumps(all_fields, indent=2))
    verify_raw, lat3 = _raw_llm_call(token, model, verify_prompt, max_tokens=500, temperature=0.0)
    total_latency += lat3

    try:
        verify_result = json.loads(verify_raw)
        corrections = verify_result.get("corrections", [])
        for correction in corrections:
            fname = correction.get("field")
            new_val = correction.get("new_value")
            if fname and new_val and fname in gen_result.get("augmented_fields", {}):
                gen_result["augmented_fields"][fname]["value"] = new_val
                gen_result["augmented_fields"][fname]["reasoning"] += f" [corrected: {correction.get('reason', '')}]"
                gen_result["augmented_fields"][fname]["confidence"] = max(0.3, gen_result["augmented_fields"][fname].get("confidence", 0.5) - 0.1)
    except (json.JSONDecodeError, KeyError):
        pass  # verification failed to parse — use uncorrected results

    return gen_result, total_latency


# ---------------------------------------------------------------------------
# Heuristic scoring
# ---------------------------------------------------------------------------

KNOWN_MODELS = {
    "gpt", "bert", "resnet", "llama", "vit", "yolo", "t5", "clip", "gemma",
    "mistral", "qwen", "falcon", "phi", "whisper", "sam", "dino", "wav2vec",
    "mobilenet", "efficientnet", "inception", "densenet", "unet", "blip",
    "llava", "stable diffusion", "sdxl", "bark", "opt", "gpt2", "gpt-2",
    "xgboost", "lightgbm", "catboost", "randomforest", "random forest",
    "gcn", "gnn", "gat",
}
KNOWN_INPUT_TYPES = {"image", "text", "tabular", "audio", "video", "multimodal"}

VALID_CATEGORIES = {
    "classification", "regression", "clustering", "anomaly detection",
    "dimensionality reduction", "reinforcement learning", "natural language processing",
    "computer vision", "recommendation systems", "time series forecasting",
    "graph learning", "graph neural networks", "generative modeling",
    "transfer learning", "self-supervised learning", "semi-supervised learning",
    "unsupervised learning", "causal inference", "multi-task learning",
    "metric learning", "density estimation", "multi-label classification",
    "ranking", "structured prediction", "neural architecture search",
    "sequence modeling", "embedding learning", "other",
}
VALID_FRAMEWORKS = {"sklearn", "tensorflow", "pytorch", "other"}
VALID_MODEL_TYPES = {"cnn", "decision_tree", "dnn", "rnn", "svm", "kmeans", "llm", "random_forest", "lstm", "gnn", "other"}


def heuristic_score(field_name: str, value, context: dict) -> float:
    if value is None or (isinstance(value, str) and not value.strip()):
        return 0.0
    if isinstance(value, bool):
        return 0.9
    s = str(value).strip()

    if field_name == "keywords":
        terms = [t.strip() for t in s.split(",") if t.strip()]
        return 1.0 if len(terms) >= 2 and all(len(t) < 40 for t in terms) else 0.4
    if field_name == "short_description" or field_name == "ai_model_description":
        if len(s) < 20 or len(s) > 200:
            return 0.3
        return 0.9
    if field_name == "full_description":
        if len(s) < 30: return 0.2
        return 0.9 if len(s) > 50 else 0.5
    if field_name == "citation":
        return 0.9 if "@" in s else 0.4
    if field_name == "foundational_model":
        return 1.0 if any(m in s.lower() for m in KNOWN_MODELS) else 0.3
    if field_name == "input_type":
        return 1.0 if s.lower().strip() in KNOWN_INPUT_TYPES else 0.3
    if field_name == "category":
        return 1.0 if s.lower() in VALID_CATEGORIES else 0.3
    if field_name == "ai_model_framework":
        return 1.0 if s.lower() in VALID_FRAMEWORKS else 0.3
    if field_name == "ai_model_model_type":
        return 1.0 if s.lower() in VALID_MODEL_TYPES else 0.3
    if field_name == "input_data":
        return 0.9 if s.startswith("http") else (0.5 if s.lower() != "null" else 0.0)
    if field_name == "output_data":
        return 0.9 if 1 < len(s) < 100 else 0.3
    if field_name in ("documentation", "ai_model_location"):
        return 0.9 if s.startswith("http") else 0.3
    if field_name in ("name", "author", "ai_model_owner", "ai_model_license"):
        return 0.9 if 1 < len(s) < 100 else 0.3
    if field_name in ("version", "ai_model_version"):
        return 0.9 if s else 0.3
    if field_name in ("is_private", "is_gated"):
        return 0.9
    if field_name == "ai_model_test_accuracy":
        return 0.0
    # Datasheet-specific validators
    if field_name == "title":
        return 0.9 if 2 < len(s) < 200 else 0.3
    if field_name == "description":
        title = context.get("title") or ""
        if len(s) < 20: return 0.2
        if s.strip().lower() == title.strip().lower(): return 0.1
        return 0.9 if len(s) > 50 else 0.5
    if field_name == "subjects":
        terms = [t.strip() for t in s.split(",") if t.strip()]
        return 1.0 if len(terms) >= 2 and all(len(t) < 50 for t in terms) else 0.4
    if field_name == "resource_type_general":
        return 1.0 if s in DATACITE_TYPES else 0.2
    if field_name == "resource_type":
        return 0.9 if 5 < len(s) < 200 else 0.3
    if field_name == "publication_year":
        try:
            year = int(value)
            return 1.0 if 1990 <= year <= 2030 else 0.2
        except (ValueError, TypeError):
            return 0.0
    if field_name == "format":
        return 1.0 if any(fmt in s.lower() for fmt in KNOWN_FORMATS) else 0.3
    if field_name == "size":
        return 0.9 if len(s) > 1 else 0.3
    if field_name in ("creator", "publisher", "license"):
        return 0.9 if 1 < len(s) < 100 else 0.3
    return 0.5


def composite(llm_confidence: float, heur: float) -> float:
    return round(0.4 * llm_confidence + 0.6 * heur, 3)

# ---------------------------------------------------------------------------
# Accuracy measurement
# ---------------------------------------------------------------------------

def semantic_overlap(gt: str, aug: str) -> float:
    if not gt or not aug:
        return 0.0
    gt_tokens = set(gt.lower().replace(",", " ").replace("-", " ").split())
    aug_tokens = set(aug.lower().replace(",", " ").replace("-", " ").split())
    if not gt_tokens:
        return 0.0
    return len(gt_tokens & aug_tokens) / max(len(gt_tokens), len(aug_tokens))


def exact_match_check(gt: str | None, aug: str | None) -> bool:
    if gt is None and aug is None:
        return True
    if gt is None or aug is None:
        return False
    return gt.strip().lower() == aug.strip().lower()

# ---------------------------------------------------------------------------
# Process one card
# ---------------------------------------------------------------------------

def process_card(token: str, model: str, entry: dict, use_readme: bool = True, deterministic_readme: bool = False, method: str = "baseline") -> CardResult:
    card_id = entry["id"]
    repo_id = entry["repo_id"]
    asset_type = entry.get("asset_type", "model_card")
    hf_card = entry["hf_api_response"]
    patra_gt = entry["patra_ground_truth"]
    readme_body = entry.get("readme_body") or "" if use_readme else ""
    augmentation_method = method  # snapshot — `method` is shadowed later in the per-field loop

    if method in ("structured", "fewshot", "cot", "experiment") and asset_type == "model_card":
        target_fields = PATRA_TARGET_FIELDS
        required_fields = MC_REQUIRED_FIELDS
        extracted = structured_extraction(hf_card, readme_body)
        if method in ("structured", "experiment"):
            never_llm = set(PATRA_TARGET_FIELDS.keys()) - {"category", "input_type", "ai_model_model_type"}
        else:
            never_llm = {"citation"}
        llm_readme = readme_body
    elif asset_type == "datasheet":
        target_fields = PATRA_DATASHEET_FIELDS
        required_fields = DS_REQUIRED_FIELDS
        extracted = extract_from_hf_dataset_json(hf_card)
        if use_readme:
            extracted = extract_from_readme(readme_body, extracted)
            if deterministic_readme or method == "structured":
                extracted = extract_descriptions_from_readme(readme_body, extracted)
        never_llm = set()
        llm_readme = readme_body
    else:
        target_fields = PATRA_TARGET_FIELDS
        required_fields = MC_REQUIRED_FIELDS
        extracted = extract_from_hf_json(hf_card)
        if use_readme:
            extracted = extract_from_readme(readme_body, extracted)
            if deterministic_readme:
                extracted = extract_descriptions_from_readme(readme_body, extracted)
        never_llm = {"citation", "ai_model_test_accuracy"}
        llm_readme = "" if deterministic_readme else readme_body

    tier1_filled = sum(1 for v in extracted.values() if v is not None)

    missing_fields = {k: target_fields[k] for k, v in extracted.items() if v is None and k not in never_llm}
    tier2_filled = 0
    llm_results = {}
    latency = 0

    if missing_fields:
        try:
            if method == "fewshot":
                llm_response, latency = call_llm_fewshot(token, model, hf_card, extracted, missing_fields, llm_readme)
            elif method == "cot":
                llm_response, latency = call_llm_cot(token, model, hf_card, extracted, missing_fields, llm_readme)
            else:
                llm_response, latency = call_llm(token, model, hf_card, extracted, missing_fields, llm_readme)
            llm_results = llm_response.get("augmented_fields") or {}
            tier2_filled = sum(1 for f in llm_results.values() if f.get("value") is not None)
        except Exception as e:
            print(f"    LLM warning for {repo_id}: {e} — continuing with deterministic fields only")

    fields = []
    for fname in target_fields:
        tier1_val = extracted.get(fname)

        if tier1_val is not None:
            aug_val = tier1_val
            conf = 1.0
            reasoning = "Deterministic extraction from HF API JSON"
            method = "deterministic"
            if fname == "citation" and "@" in str(tier1_val):
                method = "readme_bibtex"
        else:
            suggestion = llm_results.get(fname, {})
            aug_val = suggestion.get("value")
            conf = float(suggestion.get("confidence", 0))
            reasoning = suggestion.get("reasoning", "")
            method = "llm_with_readme" if readme_body else "llm"

        if asset_type == "model_card" and fname.startswith("ai_model_"):
            gt_val = (patra_gt.get("ai_model") or {}).get(fname.removeprefix("ai_model_"))
        else:
            gt_val = patra_gt.get(fname)

        aug_str = str(aug_val) if aug_val is not None else ""
        gt_str = str(gt_val) if gt_val is not None else ""

        heur = heuristic_score(fname, aug_val, patra_gt)
        comp = composite(conf, heur)
        attr_conf = compute_attribute_confidence(fname, method, aug_val, conf)
        em = exact_match_check(gt_str or None, aug_str or None)
        so = semantic_overlap(gt_str, aug_str)

        fields.append(FieldResult(
            field_name=fname, ground_truth=gt_str or None,
            augmented_value=aug_str or None, confidence=conf,
            heuristic_score=heur, composite_score=comp,
            exact_match=em, semantic_overlap=so,
            reasoning=reasoning, extraction_method=method,
            source_score=0.0, attribute_confidence=attr_conf,
        ))

    n_fields = len(target_fields)
    filled = sum(1 for f in fields if f.augmented_value)
    completeness = round(filled / n_fields, 3) if n_fields else 0.0
    req_filled = sum(1 for f in fields if f.field_name in required_fields and f.augmented_value)
    sufficiency = round(req_filled / len(required_fields), 3) if required_fields else 0.0
    overall_conf = round(sum(f.attribute_confidence for f in fields) / n_fields, 3) if n_fields else 0.0

    # M4 runtime augmentation — additive; runtime fields are appended to `fields`
    # but excluded from static completeness / sufficiency / overall_conf computed above.
    if augmentation_method == "experiment" and asset_type == "model_card":
        fields.extend(_run_experiment_runtime_for_card(token, model, entry, extracted))

    return CardResult(
        card_id=card_id, repo_id=repo_id, asset_type=asset_type,
        domain=entry["domain"], model_used=model,
        tier1_filled=tier1_filled, tier2_filled=tier2_filled,
        fields=fields, llm_latency_ms=latency,
        completeness=completeness, sufficiency=sufficiency,
        overall_confidence=overall_conf,
    )


def _run_experiment_runtime_for_card(token: str, model: str, entry: dict, extracted: dict) -> list[FieldResult]:
    """M4 runtime augmentation for one model card. Returns FieldResult list for the 9 runtime fields."""
    try:
        from augment_runtime import MC_RUNTIME_FIELDS, run_experiment_augmentation, value_to_str
    except ImportError as e:
        print(f"    M4 skipped: augment_runtime import failed: {e}")
        return []

    category = extracted.get("category") or entry.get("domain") or ""
    input_type = extracted.get("input_type") or ""
    name = extracted.get("name") or entry.get("repo_id", "").split("/")[-1]

    def _llm(prompt: str, max_tokens: int, temperature: float) -> tuple[str, int]:
        return _raw_llm_call(token, model, prompt, max_tokens=max_tokens, temperature=temperature)

    rt = run_experiment_augmentation(
        hf_card_id=entry["id"],
        name=name,
        category=category,
        input_type=input_type,
        llm_call=_llm,
    )

    results: list[FieldResult] = []
    for fname in MC_RUNTIME_FIELDS:
        value = rt.values.get(fname)
        aug_str = value_to_str(value)
        conf = rt.confidences.get(fname, 0.0)
        reasoning = rt.reasoning.get(fname, "")
        method_tag = "experiment_llm" if fname in {"runtime_typical_deployment_context", "runtime_known_failure_modes"} else "experiment_deterministic"
        results.append(FieldResult(
            field_name=fname, ground_truth=None,
            augmented_value=aug_str, confidence=conf,
            heuristic_score=0.0, composite_score=conf,
            exact_match=False, semantic_overlap=0.0,
            reasoning=reasoning, extraction_method=method_tag,
            source_score=0.0, attribute_confidence=conf,
        ))
    return results

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_card_result(r: CardResult):
    print(f"\n{'='*80}")
    tag = "MC" if r.asset_type == "model_card" else "DS"
    print(f"  [{r.card_id:2d}] [{tag}] {r.repo_id}")
    print(f"       domain={r.domain}  tier1={r.tier1_filled}  tier2={r.tier2_filled}  model={r.model_used}  latency={r.llm_latency_ms}ms")
    print(f"       Completeness={r.completeness:.2f}  Sufficiency={r.sufficiency:.2f}  Overall Confidence={r.overall_confidence:.2f}")
    print(f"{'='*80}")
    if r.error:
        print(f"  ERROR: {r.error}")
        return
    for f in r.fields:
        aug_display = (f.augmented_value or "")[:50]
        gt_display = (f.ground_truth or "")[:50]
        em_icon = "Y" if f.exact_match else "N"
        src_tag = f.extraction_method[:4]
        print(f"  {f.field_name:<22s} [{src_tag}]  attrConf={f.attribute_confidence:.2f}  exact={em_icon}  overlap={f.semantic_overlap:.2f}")
        print(f"    GT:  {gt_display}")
        print(f"    AUG: {aug_display}")


def print_summary(results: list[CardResult]):
    ok_results = [r for r in results if r.error is None]
    if not ok_results:
        print("\nNo results.")
        return

    all_fields = [f for r in ok_results for f in r.fields]

    # Split by asset type
    mc_results = [r for r in ok_results if r.asset_type == "model_card"]
    ds_results = [r for r in ok_results if r.asset_type == "datasheet"]

    print(f"\n{'='*80}")
    print(f"  SUMMARY")
    print(f"{'='*80}")
    print(f"  Records processed:     {len(results)} ({sum(1 for r in results if r.error)} errors)")
    print(f"  Total fields scored:   {len(all_fields)}")

    # New metrics by asset type
    print(f"\n  --- Metrics by Record Type ---")
    print(f"  {'Metric':<22s}  {'Model Cards':>14s}  {'Datasheets':>14s}  {'All':>14s}")
    print(f"  {'-'*22}  {'-'*14}  {'-'*14}  {'-'*14}")

    def _fmt_avg(records, attr):
        if not records: return "—"
        avg = sum(getattr(r, attr) for r in records) / len(records)
        return f"{avg:.3f}"

    print(f"  {'Completeness':<22s}  {_fmt_avg(mc_results, 'completeness'):>14s}  {_fmt_avg(ds_results, 'completeness'):>14s}  {_fmt_avg(ok_results, 'completeness'):>14s}")
    print(f"  {'Sufficiency':<22s}  {_fmt_avg(mc_results, 'sufficiency'):>14s}  {_fmt_avg(ds_results, 'sufficiency'):>14s}  {_fmt_avg(ok_results, 'sufficiency'):>14s}")
    print(f"  {'Overall Confidence':<22s}  {_fmt_avg(mc_results, 'overall_confidence'):>14s}  {_fmt_avg(ds_results, 'overall_confidence'):>14s}  {_fmt_avg(ok_results, 'overall_confidence'):>14s}")

    avg_latency = sum(r.llm_latency_ms for r in ok_results) / max(len(ok_results), 1)
    print(f"\n  Avg LLM latency:       {avg_latency:.0f}ms")

    def _print_field_table(label, records, field_dict):
        if not records:
            return
        flds_all = [f for r in records for f in r.fields]
        by_f: dict[str, list[FieldResult]] = {}
        for f in flds_all:
            by_f.setdefault(f.field_name, []).append(f)
        print(f"\n  --- Per-Field: {label} ---")
        print(f"  {'Field':<22s}  {'Src':>5s}  {'AttrConf':>8s}  {'Exact%':>6s}  {'Overlap':>7s}  {'Cover%':>6s}")
        print(f"  {'-'*22}  {'-'*5}  {'-'*8}  {'-'*6}  {'-'*7}  {'-'*6}")
        for fname in field_dict:
            fl = by_f.get(fname, [])
            if not fl:
                continue
            n = len(fl)
            src = fl[0].extraction_method[:5] if all(f.extraction_method == fl[0].extraction_method for f in fl) else "mixed"
            avg_ac = sum(f.attribute_confidence for f in fl) / n
            exact_pct = 100 * sum(1 for f in fl if f.exact_match) / n
            avg_overlap = sum(f.semantic_overlap for f in fl) / n
            cover_pct = 100 * sum(1 for f in fl if f.augmented_value) / n
            print(f"  {fname:<22s}  {src:>5s}  {avg_ac:>8.2f}  {exact_pct:>5.0f}%  {avg_overlap:>7.2f}  {cover_pct:>5.0f}%")

    _print_field_table("Model Cards", mc_results, PATRA_TARGET_FIELDS)
    _print_field_table("Datasheets", ds_results, PATRA_DATASHEET_FIELDS)

    # Per-domain
    by_domain: dict[str, list[CardResult]] = {}
    for r in ok_results:
        by_domain.setdefault(r.domain, []).append(r)
    print(f"\n  --- Per-Domain ---")
    print(f"  {'Domain':<12s}  {'N':>3s}  {'Complete':>8s}  {'Suffic':>8s}  {'OvrlConf':>8s}")
    print(f"  {'-'*12}  {'-'*3}  {'-'*8}  {'-'*8}  {'-'*8}")
    for domain in ["cv", "nlp", "audio", "tabular", "multimodal", "scientific"]:
        recs = by_domain.get(domain, [])
        if not recs: continue
        n = len(recs)
        avg_c = sum(r.completeness for r in recs) / n
        avg_s = sum(r.sufficiency for r in recs) / n
        avg_o = sum(r.overall_confidence for r in recs) / n
        print(f"  {domain:<12s}  {n:>3d}  {avg_c:>8.3f}  {avg_s:>8.3f}  {avg_o:>8.3f}")

    errors = [r for r in results if r.error]
    if errors:
        print(f"\n  --- Errors ---")
        for r in errors:
            print(f"  [{r.card_id}] {r.repo_id}: {r.error}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="default", help="Label for this run in metrics CSV")
    parser.add_argument("--method", default="baseline", choices=["baseline", "structured", "fewshot", "cot", "experiment"], help="Augmentation method")
    parser.add_argument("--no-readme", action="store_true", help="Skip README (for baseline A)")
    parser.add_argument("--deterministic-readme", action="store_true", help="B1 deterministic README")
    parser.add_argument("--dataset", default=None, help="Dataset JSON file (default: synthetic_hf_cards.json)")
    args = parser.parse_args()

    print(f"=== Patra Data Augmentation POC v2 — Phase: {args.phase} ===\n")

    dataset_path = Path(__file__).parent / "data" / "inputs" / (args.dataset or "synthetic_hf_cards.json")
    if not dataset_path.exists():
        print(f"ERROR: {dataset_path} not found.")
        sys.exit(1)
    dataset = json.loads(dataset_path.read_text())
    mc_count = sum(1 for e in dataset if e.get("asset_type") == "model_card")
    ds_count = sum(1 for e in dataset if e.get("asset_type") == "datasheet")
    print(f"Loaded {len(dataset)} entries ({mc_count} model cards, {ds_count} datasheets) from {dataset_path.name}\n")

    print("Step 1: Authenticate")
    token = fetch_tapis_token()
    print(f"  Token acquired.\n")

    resp = httpx.get(f"{LITELLM_BASE}/models", headers={"X-Tapis-Token": token}, timeout=15)
    resp.raise_for_status()
    models = [m["id"] for m in resp.json().get("data", []) if "whisper" not in m.get("id", "").lower()]
    model = next((m for m in models if "llama" in m.lower()), models[0])
    print(f"  LLM model: {model}\n")

    print("Step 2: Two-tier augmentation (HF API JSON → Patra)")
    phase_slug = args.phase.replace(" ", "_").replace(":", "").lower()
    augmented_cards_dir = Path(__file__).parent / "data" / "outputs" / "augmented_cards"
    augmented_cards_dir.mkdir(exist_ok=True)
    augmented_ds_dir = Path(__file__).parent / "data" / "outputs" / "augmented_datasheets"
    augmented_ds_dir.mkdir(exist_ok=True)

    results: list[CardResult] = []
    for entry in dataset:
        r = process_card(token, model, entry, use_readme=not args.no_readme, deterministic_readme=args.deterministic_readme, method=args.method)
        print_card_result(r)
        results.append(r)

        # Write augmented output file
        repo_id = entry["repo_id"]
        slug = repo_id.replace("/", "_").lower()
        asset_type = entry.get("asset_type", "model_card")

        if asset_type == "datasheet":
            aug_file = augmented_ds_dir / f"{entry['id']:02d}_{slug}.json"
            ds_fields = {}
            for f in r.fields:
                if f.augmented_value:
                    ds_fields[f.field_name] = f.augmented_value
            with open(aug_file, "w") as fh:
                json.dump(ds_fields, fh, indent=2)
        else:
            aug_file = augmented_cards_dir / f"{entry['id']:02d}_{slug}.json"
            ai_model_fields = {}
            card_fields = {}
            runtime_fields = {}
            for f in r.fields:
                val = f.augmented_value
                if not val:
                    continue
                fname = f.field_name
                if fname.startswith("runtime_"):
                    runtime_key = fname.removeprefix("runtime_")
                    # Restore native types for fields that were JSON-stringified for storage.
                    if runtime_key == "expected_f1_range":
                        try:
                            val = json.loads(val)
                        except (ValueError, TypeError):
                            continue
                    elif runtime_key in ("expected_latency_ms", "expected_total_power_w"):
                        try:
                            val = float(val)
                        except (ValueError, TypeError):
                            continue
                    elif runtime_key == "recommended_min_ram_mb":
                        try:
                            val = int(val)
                        except (ValueError, TypeError):
                            continue
                    runtime_fields[runtime_key] = val
                elif fname.startswith("ai_model_"):
                    key = fname.removeprefix("ai_model_")
                    if key == "test_accuracy":
                        try:
                            val = float(val)
                        except (ValueError, TypeError):
                            continue
                    ai_model_fields[key] = val
                elif fname in ("is_private", "is_gated"):
                    card_fields[fname] = val in ("True", "true", True)
                else:
                    card_fields[fname] = val
            if "name" in card_fields:
                ai_model_fields.setdefault("name", card_fields["name"])
            if ai_model_fields:
                card_fields["ai_model"] = ai_model_fields
            if runtime_fields:
                card_fields["runtime"] = runtime_fields
            with open(aug_file, "w") as fh:
                json.dump(card_fields, fh, indent=2)

    print_summary(results)

    out_path = Path(__file__).parent / "data" / "outputs" / f"results_{phase_slug}.json"
    with open(out_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2, default=str)

    # Write per-field metrics to CSV (append mode for cross-run comparison)
    csv_path = Path(__file__).parent / "data" / "outputs" / "metrics_comparison.csv"
    write_header = not csv_path.exists()
    ok_results = [r for r in results if r.error is None]
    all_fields = [f for r in ok_results for f in r.fields]
    by_field: dict[str, list[FieldResult]] = {}
    for f in all_fields:
        by_field.setdefault(f.field_name, []).append(f)

    # Pre-compute field → asset_type mapping
    field_asset_types: dict[str, set[str]] = {}
    for r in ok_results:
        for f in r.fields:
            field_asset_types.setdefault(f.field_name, set()).add(r.asset_type)

    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(["phase", "asset_type", "field", "extraction_method", "n", "attr_confidence", "composite", "exact_pct", "semantic_overlap", "coverage_pct"])
        all_target_fields = list(PATRA_TARGET_FIELDS.keys()) + list(PATRA_DATASHEET_FIELDS.keys())
        try:
            from augment_runtime import MC_RUNTIME_FIELDS
            all_target_fields += list(MC_RUNTIME_FIELDS.keys())
        except ImportError:
            pass
        for fname in dict.fromkeys(all_target_fields):  # deduplicate preserving order
            flds = by_field.get(fname, [])
            if not flds:
                continue
            n = len(flds)
            methods = {f.extraction_method for f in flds}
            method = methods.pop() if len(methods) == 1 else "mixed"
            asset_types = field_asset_types.get(fname, set())
            at = asset_types.pop() if len(asset_types) == 1 else "both"
            avg_ac = sum(f.attribute_confidence for f in flds) / n
            avg_comp = sum(f.composite_score for f in flds) / n
            exact_pct = 100 * sum(1 for f in flds if f.exact_match) / n
            avg_overlap = sum(f.semantic_overlap for f in flds) / n
            cover_pct = 100 * sum(1 for f in flds if f.augmented_value) / n
            writer.writerow([args.phase, at, fname, method, n, f"{avg_ac:.3f}", f"{avg_comp:.3f}", f"{exact_pct:.1f}", f"{avg_overlap:.2f}", f"{cover_pct:.1f}"])

    print(f"\n  Results:             {out_path}")
    print(f"  Metrics CSV:         {csv_path}")
    print(f"  Augmented cards:     poc/augmented_cards/")
    print(f"  Augmented datasheets:poc/augmented_datasheets/")


if __name__ == "__main__":
    main()
