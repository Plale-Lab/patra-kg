import argparse
import json
import math
import os
import re
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True)
class FlattenedField:
    path: str
    json_type: str
    required: bool
    fmt: str = ""
    array_depth: int = 0
    enum_values: Tuple[str, ...] = ()
    description: str = ""
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    median: Optional[float] = None
    p90: Optional[float] = None
    zero_ratio: Optional[float] = None
    negative_ratio: Optional[float] = None
    top_values: Tuple[str, ...] = ()

    @property
    def signature(self) -> Tuple[Any, ...]:
        return (
            self.path,
            self.json_type,
            self.required,
            self.fmt,
            self.array_depth,
            self.enum_values,
        )


@dataclass
class SchemaDocument:
    schema_id: str
    schema: Dict[str, Any]
    fields: List[FlattenedField]
    semantic_text: str
    signatures: Tuple[Tuple[Any, ...], ...]


@dataclass
class CandidateScore:
    schema_id: str
    vector_score: float
    exact_field_overlap: float
    semantic_alignment_score: float
    derived_support_score: float
    type_compatibility: float
    structural_score: float
    profile_compatibility: float
    deterministic_score: float
    matched_fields: List[str] = field(default_factory=list)
    missing_fields: List[str] = field(default_factory=list)
    type_conflicts: List[str] = field(default_factory=list)
    aligned_pairs: List[str] = field(default_factory=list)
    derived_support: List[str] = field(default_factory=list)
    tradeoffs: List[str] = field(default_factory=list)


@dataclass
class MatchResult:
    method: str
    winner: str
    confidence: float
    report: Dict[str, Any]


def _split_identifier(text: str) -> List[str]:
    if not text:
        return []
    text = re.sub(r"walltime", "wall time", text, flags=re.IGNORECASE)
    text = re.sub(r"runtime", "run time", text, flags=re.IGNORECASE)
    text = re.sub(r"waittime", "wait time", text, flags=re.IGNORECASE)
    text = re.sub(r"jobid", "job id", text, flags=re.IGNORECASE)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = re.sub(r"[-_/.:\\]+", " ", text)
    return [token.lower() for token in TOKEN_RE.findall(text)]


def _normalize_type(schema: Dict[str, Any]) -> str:
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        return "|".join(sorted(str(item) for item in schema_type))
    if schema_type:
        return str(schema_type)
    if "properties" in schema:
        return "object"
    if "items" in schema:
        return "array"
    if "enum" in schema:
        return "enum"
    return "unknown"


def _safe_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_top_values(profile: Dict[str, Any]) -> Tuple[str, ...]:
    normalized: List[str] = []
    for item in profile.get("top_values", []):
        if isinstance(item, (list, tuple)) and item:
            normalized.append(str(item[0]))
        else:
            normalized.append(str(item))
    return tuple(normalized)


def _bucket_ratio(value: Optional[float], prefix: str) -> List[str]:
    if value is None:
        return []
    if value == 0.0:
        label = "none"
    elif value < 0.1:
        label = "trace"
    elif value < 0.4:
        label = "low"
    elif value < 0.8:
        label = "medium"
    elif value < 1.0:
        label = "high"
    else:
        label = "all"
    return [prefix, f"{prefix}_{label}"]


def _bucket_magnitude(value: Optional[float], prefix: str) -> List[str]:
    if value is None:
        return []
    magnitude = 0 if value == 0 else int(math.floor(math.log10(abs(value) + 1.0)))
    sign = "neg" if value < 0 else "pos"
    return [prefix, f"{prefix}_{sign}", f"{prefix}_mag_{magnitude}"]


def _profile_tokens(field: FlattenedField) -> List[str]:
    tokens: List[str] = []
    if field.median is not None or field.p90 is not None or field.maximum is not None:
        tokens.append("profiled_numeric_field")
        tokens.extend(_bucket_magnitude(field.median, "median"))
        tokens.extend(_bucket_magnitude(field.p90, "p90"))
        tokens.extend(_bucket_magnitude(field.maximum, "max"))
        tokens.extend(_bucket_ratio(field.zero_ratio, "zero_ratio"))
        tokens.extend(_bucket_ratio(field.negative_ratio, "negative_ratio"))
        if field.minimum is not None and field.maximum is not None:
            tokens.append(
                "constant_field" if abs(field.maximum - field.minimum) < 1e-9 else "variable_field"
            )
    for value in field.top_values:
        tokens.extend(_split_identifier(value))
    return tokens


def _flatten_schema(
    schema: Dict[str, Any],
    path: str = "$",
    required: bool = True,
    array_depth: int = 0,
) -> List[FlattenedField]:
    normalized_type = _normalize_type(schema)
    enum_values = tuple(sorted(str(item) for item in schema.get("enum", [])))
    profile = schema.get("x-profile", {}) if isinstance(schema.get("x-profile"), dict) else {}
    field = FlattenedField(
        path=path,
        json_type=normalized_type,
        required=required,
        fmt=str(schema.get("format", "")),
        array_depth=array_depth,
        enum_values=enum_values,
        description=str(schema.get("description", "")),
        minimum=_safe_float(schema.get("minimum", profile.get("min"))),
        maximum=_safe_float(schema.get("maximum", profile.get("max"))),
        median=_safe_float(profile.get("p50")),
        p90=_safe_float(profile.get("p90")),
        zero_ratio=_safe_float(profile.get("zero_ratio")),
        negative_ratio=_safe_float(profile.get("neg_ratio")),
        top_values=_normalize_top_values(profile),
    )
    flattened = [field]

    if normalized_type == "object" or "properties" in schema:
        required_fields = set(schema.get("required", []))
        for prop_name in sorted(schema.get("properties", {}).keys()):
            child = schema["properties"][prop_name]
            child_path = f"{path}.{prop_name}"
            flattened.extend(
                _flatten_schema(
                    child,
                    path=child_path,
                    required=prop_name in required_fields,
                    array_depth=array_depth,
                )
            )

    if normalized_type == "array" or "items" in schema:
        items = schema.get("items", {})
        flattened.extend(
            _flatten_schema(
                items,
                path=f"{path}[]",
                required=required,
                array_depth=array_depth + 1,
            )
        )

    return flattened


def _schema_to_semantic_text(fields: Iterable[FlattenedField]) -> str:
    chunks: List[str] = []
    for field in fields:
        chunks.extend(_split_identifier(field.path))
        chunks.append(field.json_type)
        if field.fmt:
            chunks.extend(_split_identifier(field.fmt))
        if field.description:
            chunks.extend(_split_identifier(field.description))
        if field.enum_values:
            for value in field.enum_values:
                chunks.extend(_split_identifier(value))
        chunks.extend(_profile_tokens(field))
    return " ".join(chunks)


def _schema_to_semantic_only_text(fields: Iterable[FlattenedField]) -> str:
    chunks: List[str] = []
    for field in fields:
        chunks.extend(_split_identifier(field.path))
        if field.description:
            chunks.extend(_split_identifier(field.description))
        if field.enum_values:
            for value in field.enum_values:
                chunks.extend(_split_identifier(value))
    return " ".join(chunks)


def _tokenize(text: str) -> Counter[str]:
    return Counter(_split_identifier(text))


def _cosine_similarity(left: Dict[str, float], right: Dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    dot = 0.0
    for token, weight in left.items():
        dot += weight * right.get(token, 0.0)
    left_norm = math.sqrt(sum(weight * weight for weight in left.values()))
    right_norm = math.sqrt(sum(weight * weight for weight in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _json_leaf_paths(fields: Iterable[FlattenedField]) -> Dict[str, FlattenedField]:
    return {
        field.path: field
        for field in fields
        if field.json_type not in {"object", "array"} and field.path != "$"
    }


NUMERIC_TYPES = {"integer", "number"}
STRING_LIKE_TYPES = {"string", "enum"}
ROLE_TOKEN_ALIASES = {
    "cpu": {"cores", "core", "processors", "processor", "allocated_processors"},
    "cores": {"cpu", "processors", "processor"},
    "processors": {"cpu", "cores", "processor"},
    "processor": {"cpu", "cores", "processors"},
    "job": {"workload", "trace"},
    "status": {"state"},
    "state": {"status"},
    "queue": {"partition"},
    "partition": {"queue"},
    "wall": {"duration"},
    "duration": {"wall"},
    "runtime": {"run_time", "run", "execution"},
}
SPECIAL_ROLE_COMPATIBILITY = {
    frozenset({"requested_walltime", "wall_time"}): 0.58,
    frozenset({"queue_wait", "wall_time"}): 0.28,
    frozenset({"queue_wait", "runtime"}): 0.16,
    frozenset({"submit_time", "interarrival_time"}): 0.18,
    frozenset({"end_time", "wall_time"}): 0.16,
}
DERIVABLE_ROLE_SUPPORT = {
    "queue_wait": [
        (
            {"wall_time", "runtime"},
            0.72,
            "Queue wait can be approximated from wall_time - run_time.",
        )
    ],
}


def _leaf_name(path: str) -> str:
    return path.split(".")[-1].replace("[]", "")


def _expanded_field_tokens(field: FlattenedField) -> set[str]:
    tokens = set(_split_identifier(_leaf_name(field.path)))
    tokens.update(_split_identifier(field.description))
    expanded = set(tokens)
    for token in list(tokens):
        expanded.update(ROLE_TOKEN_ALIASES.get(token, set()))
    return expanded


def _infer_field_roles(field: FlattenedField) -> set[str]:
    name = _leaf_name(field.path).lower()
    tokens = set(_split_identifier(name))
    description_tokens = set(_split_identifier(field.description))
    combined = tokens | description_tokens
    roles: set[str] = set()

    if name in {"u_id", "job_number", "jobid", "job_id"} or (
        "job" in combined and "id" in combined
    ):
        roles.add("job_identifier")
    if "user" in combined:
        roles.add("user_identifier")
    if "gpu" in combined:
        roles.add("gpu_request")
    if {"cpu", "core"} & combined or "processor" in combined or "processors" in combined:
        roles.add("cpu_request")
    if "node" in combined or "nodes" in combined:
        roles.add("node_request")
    if "submit" in combined and "time" in combined:
        roles.add("submit_time")
    if "start" in combined and "time" in combined:
        roles.add("start_time")
    if "end" in combined and "time" in combined:
        roles.add("end_time")
    if "state" in combined or "status" in combined:
        roles.add("job_state")
    if "qos" in combined:
        roles.add("qos")
    if "queue" in combined and ("name" in combined or "partition" in combined):
        roles.add("queue_name")
    if "queue" in combined and "wait" in combined:
        roles.add("queue_wait")
    if "interval" in combined:
        roles.add("interarrival_time")
    if ("run" in combined and "time" in combined) or "runtime" in combined:
        roles.add("runtime")
    if "wall" in combined and "time" in combined:
        roles.add("wall_time")
    if "requested_walltime" in combined or (
        "requested" in combined and ("walltime" in combined or "wall" in combined)
    ):
        roles.add("requested_walltime")
    if "requested" in combined and ("core" in combined or "cores" in combined):
        roles.add("cpu_request")
    if "requested" in combined and ("node" in combined or "nodes" in combined):
        roles.add("node_request")
    if "power" in combined:
        roles.add("power")
    return roles


def _type_similarity(left_type: str, right_type: str) -> float:
    if left_type == right_type:
        return 1.0
    if left_type in NUMERIC_TYPES and right_type in NUMERIC_TYPES:
        return 0.9
    if left_type in STRING_LIKE_TYPES and right_type in STRING_LIKE_TYPES:
        return 0.85
    return 0.2


def _token_jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


class SparseTfIdfIndex:
    def __init__(self, documents: List[SchemaDocument]):
        self.documents = documents
        self.idf = self._build_idf([_tokenize(doc.semantic_text) for doc in documents])
        self.vectors = {
            doc.schema_id: self._vectorize(_tokenize(doc.semantic_text)) for doc in documents
        }

    def _build_idf(self, counters: List[Counter[str]]) -> Dict[str, float]:
        total_docs = max(len(counters), 1)
        doc_frequency: Counter[str] = Counter()
        for counter in counters:
            doc_frequency.update(counter.keys())
        return {
            token: math.log((1 + total_docs) / (1 + freq)) + 1.0
            for token, freq in doc_frequency.items()
        }

    def _vectorize(self, counter: Counter[str]) -> Dict[str, float]:
        return {
            token: count * self.idf.get(token, 1.0)
            for token, count in counter.items()
            if token.strip()
        }

    def query(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        target_vector = self._vectorize(_tokenize(text))
        ranked = [
            (schema_id, _cosine_similarity(target_vector, vector))
            for schema_id, vector in self.vectors.items()
        ]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:top_k]


class SparseCosineRetriever:
    def __init__(self, texts_by_id: Dict[str, str]):
        self.texts_by_id = texts_by_id
        self.idf = self._build_idf([_tokenize(text) for text in texts_by_id.values()])
        self.vectors = {
            schema_id: self._vectorize(_tokenize(text))
            for schema_id, text in texts_by_id.items()
        }

    def _build_idf(self, counters: List[Counter[str]]) -> Dict[str, float]:
        total_docs = max(len(counters), 1)
        doc_frequency: Counter[str] = Counter()
        for counter in counters:
            doc_frequency.update(counter.keys())
        return {
            token: math.log((1 + total_docs) / (1 + freq)) + 1.0
            for token, freq in doc_frequency.items()
        }

    def _vectorize(self, counter: Counter[str]) -> Dict[str, float]:
        return {
            token: count * self.idf.get(token, 1.0)
            for token, count in counter.items()
            if token.strip()
        }

    def query(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        target_vector = self._vectorize(_tokenize(text))
        ranked = [
            (schema_id, _cosine_similarity(target_vector, vector))
            for schema_id, vector in self.vectors.items()
        ]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:top_k]


class LocalOpenAICompatibleLLM:
    def __init__(
        self,
        api_base: str,
        model: str,
        api_key: Optional[str] = None,
        timeout_seconds: int = 60,
    ):
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "lm-studio")
        self.timeout_seconds = timeout_seconds

    def rerank(
        self,
        target_schema: Dict[str, Any],
        candidates: List[CandidateScore],
        documents_by_id: Dict[str, SchemaDocument],
    ) -> Dict[str, Any]:
        prompt = self._build_prompt(target_schema, candidates, documents_by_id)
        last_error: Optional[Exception] = None

        for use_json_schema in (True, False):
            try:
                return self._send_request(prompt, use_json_schema=use_json_schema)
            except Exception as exc:  # noqa: BLE001
                last_error = exc

        raise RuntimeError(f"LLM rerank failed: {last_error}")

    def _send_request(self, prompt: str, use_json_schema: bool) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "temperature": 0.7,
            "top_p": 0.8,
            "presence_penalty": 1.5,
            "max_tokens": 1024,
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a schema matching agent. "
                        "Return strict JSON only, do not wrap it in markdown, "
                        "and do not include any thinking process, reasoning trace, "
                        "or <think> tags."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }
        if use_json_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "schema_match_report",
                    "schema": self._response_json_schema(),
                },
            }

        request = urllib.request.Request(
            url=f"{self.api_base}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            raw = response.read().decode("utf-8")
        parsed = json.loads(raw)
        content = parsed["choices"][0]["message"]["content"]
        return self._extract_json(content)

    def _response_json_schema(self) -> Dict[str, Any]:
        recommendation = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "schema_id": {"type": "string"},
                "rank": {"type": "integer"},
                "overall_score": {"type": "number"},
                "decision": {
                    "type": "string",
                    "enum": ["recommend", "possible", "reject"],
                },
                "summary": {"type": "string"},
                "matched_fields": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "missing_fields": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "type_conflicts": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "aligned_pairs": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "derived_support": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "tradeoffs": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": [
                "schema_id",
                "rank",
                "overall_score",
                "decision",
                "summary",
                "matched_fields",
                "missing_fields",
                "type_conflicts",
                "aligned_pairs",
                "derived_support",
                "tradeoffs",
            ],
        }
        eliminated = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "schema_id": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["schema_id", "reason"],
        }
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "winner": {"type": "string"},
                "confidence": {"type": "number"},
                "top_3": {
                    "type": "array",
                    "items": recommendation,
                    "minItems": 1,
                    "maxItems": 3,
                },
                "ranking": {
                    "type": "array",
                    "items": recommendation,
                    "minItems": 1,
                },
                "eliminated": {
                    "type": "array",
                    "items": eliminated,
                },
            },
            "required": ["winner", "confidence", "top_3", "ranking", "eliminated"],
        }

    def _extract_json(self, content: str) -> Dict[str, Any]:
        content = content.strip()
        if content.startswith("{") and content.endswith("}"):
            return json.loads(content)
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"Model did not return JSON: {content}")
        return json.loads(content[start : end + 1])

    def _build_prompt(
        self,
        target_schema: Dict[str, Any],
        candidates: List[CandidateScore],
        documents_by_id: Dict[str, SchemaDocument],
    ) -> str:
        candidate_blocks = []
        for candidate in candidates:
            doc = documents_by_id[candidate.schema_id]
            candidate_blocks.append(
                {
                    "schema_id": candidate.schema_id,
                    "vector_score": round(candidate.vector_score, 4),
                    "deterministic_score": round(candidate.deterministic_score, 4),
                    "semantic_alignment_score": round(candidate.semantic_alignment_score, 4),
                    "derived_support_score": round(candidate.derived_support_score, 4),
                    "profile_compatibility": round(candidate.profile_compatibility, 4),
                    "matched_fields": candidate.matched_fields,
                    "missing_fields": candidate.missing_fields,
                    "type_conflicts": candidate.type_conflicts,
                    "aligned_pairs": candidate.aligned_pairs,
                    "derived_support": candidate.derived_support,
                    "candidate_schema": doc.schema,
                }
            )

        instructions = {
            "task": (
                "Rank the candidate schemas for the target schema. "
                "Prefer structural compatibility over superficial term overlap."
            ),
            "output_contract": {
                "winner": "schema_id",
                "confidence": "0-1 float",
                "top_3": [
                    {
                        "schema_id": "string",
                        "rank": "integer",
                        "overall_score": "0-1 float",
                        "decision": "recommend | possible | reject",
                        "summary": "short explanation",
                        "matched_fields": ["field paths"],
                        "missing_fields": ["field paths"],
                        "type_conflicts": ["field conflict descriptions"],
                        "aligned_pairs": ["target -> candidate mappings"],
                        "derived_support": ["derivable target capabilities"],
                        "tradeoffs": ["short bullet strings"],
                    }
                ],
                "ranking": [
                    {
                        "schema_id": "string",
                        "rank": "integer",
                        "overall_score": "0-1 float",
                        "decision": "recommend | possible | reject",
                        "summary": "short explanation",
                        "matched_fields": ["field paths"],
                        "missing_fields": ["field paths"],
                        "type_conflicts": ["field conflict descriptions"],
                        "aligned_pairs": ["target -> candidate mappings"],
                        "derived_support": ["derivable target capabilities"],
                        "tradeoffs": ["short bullet strings"],
                    }
                ],
                "eliminated": [
                    {"schema_id": "string", "reason": "short explanation"}
                ],
            },
        }

        return json.dumps(
            {
                "instructions": instructions,
                "target_schema": target_schema,
                "candidates": candidate_blocks,
            },
            ensure_ascii=False,
            indent=2,
        )


class HybridSchemaMatcher:
    def __init__(
        self,
        schema_records: List[Dict[str, Any]],
        llm_client: Optional[LocalOpenAICompatibleLLM] = None,
    ):
        self.documents = [self._build_document(record) for record in schema_records]
        self.documents_by_id = {doc.schema_id: doc for doc in self.documents}
        self.vector_index = SparseTfIdfIndex(self.documents)
        self.semantic_only_index = SparseCosineRetriever(
            {
                doc.schema_id: _schema_to_semantic_only_text(doc.fields)
                for doc in self.documents
            }
        )
        self.llm_client = llm_client

    def _build_document(self, record: Dict[str, Any]) -> SchemaDocument:
        schema_id = record["id"]
        schema = record["schema"]
        fields = _flatten_schema(schema)
        return SchemaDocument(
            schema_id=schema_id,
            schema=schema,
            fields=fields,
            semantic_text=_schema_to_semantic_text(fields),
            signatures=tuple(sorted(field.signature for field in fields)),
        )

    def build_target_document(self, target_schema: Dict[str, Any]) -> SchemaDocument:
        return self._build_document({"id": "__target__", "schema": target_schema})

    def get_document(self, schema_id: str) -> SchemaDocument:
        return self.documents_by_id[schema_id]

    def match_schema(self, target_schema: Dict[str, Any], top_k: int = 10) -> MatchResult:
        target_doc = self.build_target_document(target_schema)
        exact_match = self._find_exact_match(target_doc)
        if exact_match:
            exact_candidate = {
                "schema_id": exact_match.schema_id,
                "rank": 1,
                "overall_score": 1.0,
                "decision": "recommend",
                "summary": "Exact field, type, and format match.",
                "matched_fields": list(_json_leaf_paths(target_doc.fields).keys()),
                "missing_fields": [],
                "type_conflicts": [],
                "aligned_pairs": [],
                "derived_support": [],
                "tradeoffs": [],
            }
            report = {
                "stage": "exact_match",
                "winner": exact_match.schema_id,
                "confidence": 1.0,
                "top_3": [exact_candidate],
                "ranking": [exact_candidate],
                "eliminated": [],
            }
            return MatchResult(
                method="deterministic_exact_match",
                winner=exact_match.schema_id,
                confidence=1.0,
                report=report,
            )

        candidate_scores = self._vector_recall(target_doc, top_k=top_k)
        reranked_report = self._agentic_rerank(target_doc, candidate_scores)
        return MatchResult(
            method=reranked_report.get("stage", "hybrid_vector_plus_agent"),
            winner=reranked_report["winner"],
            confidence=float(reranked_report.get("confidence", 0.0)),
            report=reranked_report,
        )

    def pure_cosine_match_schema(
        self,
        target_schema: Dict[str, Any],
        top_k: int = 10,
    ) -> MatchResult:
        target_doc = self.build_target_document(target_schema)
        ranked = self.semantic_only_index.query(
            _schema_to_semantic_only_text(target_doc.fields),
            top_k=top_k,
        )
        candidate_scores = [
            self._score_candidate(target_doc, self.documents_by_id[schema_id], vector_score)
            for schema_id, vector_score in ranked
        ]

        ranking = []
        for rank, candidate in enumerate(candidate_scores, start=1):
            ranking.append(
                {
                    "schema_id": candidate.schema_id,
                    "rank": rank,
                    "overall_score": round(candidate.vector_score, 4),
                    "decision": self._decision_from_score(
                        candidate.vector_score,
                        recommend_threshold=0.25,
                        possible_threshold=0.05,
                    ),
                    "summary": self._build_summary(candidate),
                    "matched_fields": candidate.matched_fields,
                    "missing_fields": candidate.missing_fields,
                    "type_conflicts": candidate.type_conflicts,
                    "aligned_pairs": candidate.aligned_pairs,
                    "derived_support": candidate.derived_support,
                    "tradeoffs": candidate.tradeoffs,
                }
            )

        report = {
            "stage": "pure_cosine_vector_only",
            "winner": candidate_scores[0].schema_id if candidate_scores else "",
            "confidence": candidate_scores[0].vector_score if candidate_scores else 0.0,
            "top_3": ranking[:3],
            "ranking": ranking,
            "eliminated": [
                {
                    "schema_id": candidate.schema_id,
                    "reason": self._build_summary(candidate),
                }
                for candidate in candidate_scores[3:]
            ],
        }
        return MatchResult(
            method="pure_cosine_vector_only",
            winner=report["winner"],
            confidence=float(report["confidence"]),
            report=report,
        )

    def _find_exact_match(self, target_doc: SchemaDocument) -> Optional[SchemaDocument]:
        matches = [doc for doc in self.documents if doc.signatures == target_doc.signatures]
        if len(matches) == 1:
            return matches[0]
        return None

    def _vector_recall(self, target_doc: SchemaDocument, top_k: int) -> List[CandidateScore]:
        ranked = self.vector_index.query(target_doc.semantic_text, top_k=top_k)
        candidate_scores = []
        for schema_id, vector_score in ranked:
            doc = self.documents_by_id[schema_id]
            candidate_scores.append(self._score_candidate(target_doc, doc, vector_score))
        candidate_scores.sort(key=lambda item: item.deterministic_score, reverse=True)
        return candidate_scores

    def _role_compatibility(
        self,
        target_roles: set[str],
        candidate_roles: set[str],
    ) -> Tuple[float, str]:
        direct = sorted(target_roles & candidate_roles)
        if direct:
            return 1.0, f"shared role={direct[0]}"

        best_score = 0.0
        best_reason = ""
        for target_role in target_roles:
            for candidate_role in candidate_roles:
                score = SPECIAL_ROLE_COMPATIBILITY.get(
                    frozenset({target_role, candidate_role}),
                    0.0,
                )
                if score > best_score:
                    best_score = score
                    best_reason = f"related roles={target_role}->{candidate_role}"
        return best_score, best_reason

    def _semantic_pair_score(
        self,
        target_field: FlattenedField,
        candidate_field: FlattenedField,
    ) -> Tuple[float, str]:
        target_tokens = _expanded_field_tokens(target_field)
        candidate_tokens = _expanded_field_tokens(candidate_field)
        token_score = _token_jaccard(target_tokens, candidate_tokens)
        role_score, role_reason = self._role_compatibility(
            _infer_field_roles(target_field),
            _infer_field_roles(candidate_field),
        )
        type_score = _type_similarity(target_field.json_type, candidate_field.json_type)
        score = 0.50 * role_score + 0.35 * token_score + 0.15 * type_score

        reasons: List[str] = []
        if role_reason:
            reasons.append(role_reason)
        if token_score > 0.0:
            reasons.append(f"token_overlap={token_score:.2f}")
        reasons.append(f"type_similarity={type_score:.2f}")
        return score, ", ".join(reasons)

    def _semantic_alignments(
        self,
        target_fields: Dict[str, FlattenedField],
        candidate_fields: Dict[str, FlattenedField],
        matched_fields: List[str],
    ) -> List[Tuple[str, str, float, str]]:
        exact_targets = set(matched_fields)
        exact_candidates = set(matched_fields)
        pair_candidates: List[Tuple[float, str, str, str]] = []

        for target_path, target_field in target_fields.items():
            if target_path in exact_targets:
                continue
            for candidate_path, candidate_field in candidate_fields.items():
                if candidate_path in exact_candidates:
                    continue
                score, rationale = self._semantic_pair_score(target_field, candidate_field)
                if score >= 0.45:
                    pair_candidates.append(
                        (score, target_path, candidate_path, rationale)
                    )

        pair_candidates.sort(key=lambda item: item[0], reverse=True)
        aligned: List[Tuple[str, str, float, str]] = []
        used_targets: set[str] = set()
        used_candidates: set[str] = set()
        for score, target_path, candidate_path, rationale in pair_candidates:
            if target_path in used_targets or candidate_path in used_candidates:
                continue
            aligned.append((target_path, candidate_path, score, rationale))
            used_targets.add(target_path)
            used_candidates.add(candidate_path)
        return aligned

    def _derived_support(
        self,
        target_fields: Dict[str, FlattenedField],
        candidate_fields: Dict[str, FlattenedField],
        covered_targets: set[str],
    ) -> Tuple[Dict[str, float], List[str]]:
        role_to_paths: Dict[str, List[str]] = {}
        for path, field in candidate_fields.items():
            for role in _infer_field_roles(field):
                role_to_paths.setdefault(role, []).append(path)

        support_scores: Dict[str, float] = {}
        support_notes: List[str] = []
        for target_path, target_field in target_fields.items():
            if target_path in covered_targets:
                continue
            for target_role in _infer_field_roles(target_field):
                for required_roles, score, message in DERIVABLE_ROLE_SUPPORT.get(target_role, []):
                    if not required_roles.issubset(role_to_paths):
                        continue
                    support_scores[target_path] = max(
                        support_scores.get(target_path, 0.0),
                        score,
                    )
                    candidate_refs = ", ".join(
                        sorted(role_to_paths[role][0] for role in sorted(required_roles))
                    )
                    support_notes.append(
                        f"{target_path} => {message} Supported by {candidate_refs}."
                    )
        return support_scores, support_notes

    def _score_candidate(
        self,
        target_doc: SchemaDocument,
        candidate_doc: SchemaDocument,
        vector_score: float,
    ) -> CandidateScore:
        target_fields = _json_leaf_paths(target_doc.fields)
        candidate_fields = _json_leaf_paths(candidate_doc.fields)
        matched_fields = sorted(set(target_fields).intersection(candidate_fields))
        semantic_alignments = self._semantic_alignments(
            target_fields,
            candidate_fields,
            matched_fields,
        )
        aligned_target_paths = {item[0] for item in semantic_alignments}
        covered_targets = set(matched_fields) | aligned_target_paths
        derived_support_scores, derived_support_notes = self._derived_support(
            target_fields,
            candidate_fields,
            covered_targets,
        )
        covered_targets |= set(derived_support_scores)
        missing_fields = sorted(set(target_fields).difference(covered_targets))

        type_conflicts: List[str] = []
        for path in matched_fields:
            target_field = target_fields[path]
            candidate_field = candidate_fields[path]
            if target_field.json_type != candidate_field.json_type:
                type_conflicts.append(
                    f"{path}: target={target_field.json_type}, candidate={candidate_field.json_type}"
                )
                continue
            if target_field.fmt != candidate_field.fmt and (
                target_field.fmt or candidate_field.fmt
            ):
                type_conflicts.append(
                    f"{path}: target_format={target_field.fmt or 'n/a'}, "
                    f"candidate_format={candidate_field.fmt or 'n/a'}"
                )

        aligned_pairs: List[str] = []
        supported_field_scores: List[float] = []
        for path in matched_fields:
            supported_field_scores.append(1.0)
        for target_path, candidate_path, score, rationale in semantic_alignments:
            target_field = target_fields[target_path]
            candidate_field = candidate_fields[candidate_path]
            aligned_pairs.append(
                f"{target_path} -> {candidate_path} ({rationale}, score={score:.2f})"
            )
            supported_field_scores.append(score)
            if _type_similarity(target_field.json_type, candidate_field.json_type) < 0.75:
                type_conflicts.append(
                    f"{target_path}: semantically aligned to {candidate_path} but "
                    f"type differs ({target_field.json_type} vs {candidate_field.json_type})"
                )
            if target_field.fmt != candidate_field.fmt and target_field.fmt and candidate_field.fmt:
                type_conflicts.append(
                    f"{target_path}: aligned to {candidate_path} but format differs "
                    f"({target_field.fmt} vs {candidate_field.fmt})"
                )
        for score in derived_support_scores.values():
            supported_field_scores.append(score)

        exact_field_overlap = len(matched_fields) / max(len(target_fields), 1)
        semantic_alignment_score = sum(supported_field_scores) / max(len(target_fields), 1)
        aligned_coverage = len(covered_targets) / max(len(target_fields), 1)
        derived_support_score = sum(derived_support_scores.values()) / max(len(target_fields), 1)
        supported_pair_count = len(matched_fields) + len(semantic_alignments)
        type_compatibility = max(
            0.0,
            1.0 - (len(type_conflicts) / max(supported_pair_count, 1)),
        )
        target_paths = set(field.path for field in target_doc.fields)
        candidate_paths = set(field.path for field in candidate_doc.fields)
        structural_score = len(target_paths & candidate_paths) / max(
            len(target_paths | candidate_paths),
            1,
        )
        profile_compatibility = self._profile_compatibility(
            target_fields,
            candidate_fields,
            matched_fields,
            semantic_alignments,
        )
        has_profile_signal = profile_compatibility >= 0.0
        if has_profile_signal:
            deterministic_score = (
                0.12 * exact_field_overlap
                + 0.24 * semantic_alignment_score
                + 0.10 * aligned_coverage
                + 0.09 * vector_score
                + 0.10 * type_compatibility
                + 0.05 * structural_score
                + 0.10 * derived_support_score
                + 0.30 * profile_compatibility
            )
        else:
            profile_compatibility = 0.0
            deterministic_score = (
                0.16 * exact_field_overlap
                + 0.34 * semantic_alignment_score
                + 0.12 * aligned_coverage
                + 0.14 * vector_score
                + 0.14 * type_compatibility
                + 0.05 * structural_score
                + 0.05 * derived_support_score
            )

        tradeoffs: List[str] = []
        if aligned_pairs:
            tradeoffs.append(f"Soft-aligned {len(aligned_pairs)} semantically related fields.")
        if derived_support_notes:
            tradeoffs.append(f"Derived support available for {len(derived_support_notes)} target fields.")
        if missing_fields:
            tradeoffs.append(f"Missing {len(missing_fields)} target fields.")
        if type_conflicts:
            tradeoffs.append(f"Found {len(type_conflicts)} type or format conflicts.")
        if has_profile_signal:
            tradeoffs.append(
                f"Profile compatibility={profile_compatibility:.2f} over matched numeric/categorical fields."
            )
        if not tradeoffs:
            tradeoffs.append("Strong structural alignment with the target schema.")

        return CandidateScore(
            schema_id=candidate_doc.schema_id,
            vector_score=vector_score,
            exact_field_overlap=exact_field_overlap,
            semantic_alignment_score=semantic_alignment_score,
            derived_support_score=derived_support_score,
            type_compatibility=type_compatibility,
            structural_score=structural_score,
            profile_compatibility=profile_compatibility,
            deterministic_score=deterministic_score,
            matched_fields=matched_fields,
            missing_fields=missing_fields,
            type_conflicts=type_conflicts,
            aligned_pairs=aligned_pairs,
            derived_support=derived_support_notes,
            tradeoffs=tradeoffs,
        )

    def _profile_compatibility(
        self,
        target_fields: Dict[str, FlattenedField],
        candidate_fields: Dict[str, FlattenedField],
        matched_fields: List[str],
        semantic_alignments: List[Tuple[str, str, float, str]],
    ) -> float:
        scores: List[float] = []
        for path in matched_fields:
            target_field = target_fields[path]
            candidate_field = candidate_fields[path]
            numeric_score = self._numeric_profile_similarity(target_field, candidate_field)
            if numeric_score is not None:
                scores.append(numeric_score)
                continue
            categorical_score = self._categorical_profile_similarity(target_field, candidate_field)
            if categorical_score is not None:
                scores.append(categorical_score)
        for target_path, candidate_path, _, _ in semantic_alignments:
            target_field = target_fields[target_path]
            candidate_field = candidate_fields[candidate_path]
            numeric_score = self._numeric_profile_similarity(target_field, candidate_field)
            if numeric_score is not None:
                scores.append(numeric_score)
                continue
            categorical_score = self._categorical_profile_similarity(target_field, candidate_field)
            if categorical_score is not None:
                scores.append(categorical_score)
        if not scores:
            return -1.0
        return sum(scores) / len(scores)

    def _numeric_profile_similarity(
        self,
        target_field: FlattenedField,
        candidate_field: FlattenedField,
    ) -> Optional[float]:
        target_values = (
            target_field.median,
            target_field.p90,
            target_field.maximum,
            target_field.zero_ratio,
            target_field.negative_ratio,
        )
        candidate_values = (
            candidate_field.median,
            candidate_field.p90,
            candidate_field.maximum,
            candidate_field.zero_ratio,
            candidate_field.negative_ratio,
        )
        if not any(value is not None for value in target_values) or not any(
            value is not None for value in candidate_values
        ):
            return None

        similarities: List[float] = []
        for target_value, candidate_value in zip(
            target_values[:3],
            candidate_values[:3],
        ):
            if target_value is None or candidate_value is None:
                continue
            target_log = math.copysign(math.log1p(abs(target_value)), target_value)
            candidate_log = math.copysign(math.log1p(abs(candidate_value)), candidate_value)
            similarities.append(1.0 / (1.0 + abs(target_log - candidate_log)))

        for target_value, candidate_value in zip(
            target_values[3:],
            candidate_values[3:],
        ):
            if target_value is None or candidate_value is None:
                continue
            similarities.append(max(0.0, 1.0 - abs(target_value - candidate_value)))

        if not similarities:
            return None
        return sum(similarities) / len(similarities)

    def _categorical_profile_similarity(
        self,
        target_field: FlattenedField,
        candidate_field: FlattenedField,
    ) -> Optional[float]:
        target_values = set(target_field.top_values)
        candidate_values = set(candidate_field.top_values)
        if not target_values or not candidate_values:
            return None
        union = target_values | candidate_values
        if not union:
            return None
        return len(target_values & candidate_values) / len(union)

    def _agentic_rerank(
        self,
        target_doc: SchemaDocument,
        candidates: List[CandidateScore],
    ) -> Dict[str, Any]:
        if self.llm_client:
            shortlist = candidates[: min(len(candidates), 10)]
            try:
                report = self.llm_client.rerank(
                    target_schema=target_doc.schema,
                    candidates=shortlist,
                    documents_by_id=self.documents_by_id,
                )
                report["stage"] = "hybrid_vector_plus_agent"
                if "ranking" not in report:
                    report["ranking"] = list(report.get("top_3", []))
                return report
            except Exception as exc:  # noqa: BLE001
                fallback = self._fallback_rerank(shortlist)
                fallback["stage"] = "hybrid_vector_plus_deterministic_fallback"
                fallback["llm_error"] = str(exc)
                return fallback

        fallback = self._fallback_rerank(candidates)
        fallback["stage"] = "hybrid_vector_plus_deterministic_fallback"
        return fallback

    def _fallback_rerank(self, candidates: List[CandidateScore]) -> Dict[str, Any]:
        sorted_candidates = sorted(
            candidates,
            key=lambda item: item.deterministic_score,
            reverse=True,
        )
        top_three = []
        ranking = []
        for rank, candidate in enumerate(sorted_candidates, start=1):
            decision = self._decision_from_score(
                candidate.deterministic_score,
                recommend_threshold=0.75,
                possible_threshold=0.25,
            )
            ranking.append(
                {
                    "schema_id": candidate.schema_id,
                    "rank": rank,
                    "overall_score": round(candidate.deterministic_score, 4),
                    "decision": decision,
                    "summary": self._build_summary(candidate),
                    "matched_fields": candidate.matched_fields,
                    "missing_fields": candidate.missing_fields,
                    "type_conflicts": candidate.type_conflicts,
                    "aligned_pairs": candidate.aligned_pairs,
                    "derived_support": candidate.derived_support,
                    "tradeoffs": candidate.tradeoffs,
                }
            )
        top_three = ranking[:3]

        return {
            "winner": top_three[0]["schema_id"] if top_three else "",
            "confidence": top_three[0]["overall_score"] if top_three else 0.0,
            "top_3": top_three,
            "ranking": ranking,
            "eliminated": [
                {
                    "schema_id": candidate.schema_id,
                    "reason": self._build_summary(candidate),
                }
                for candidate in sorted_candidates[3:]
            ],
        }

    def _build_summary(self, candidate: CandidateScore) -> str:
        return (
            f"overlap={candidate.exact_field_overlap:.2f}, "
            f"semantic_alignment={candidate.semantic_alignment_score:.2f}, "
            f"derived_support={candidate.derived_support_score:.2f}, "
            f"type_compatibility={candidate.type_compatibility:.2f}, "
            f"structural_score={candidate.structural_score:.2f}, "
            f"profile_compatibility={candidate.profile_compatibility:.2f}, "
            f"vector_score={candidate.vector_score:.2f}"
        )

    def _decision_from_score(
        self,
        score: float,
        recommend_threshold: float,
        possible_threshold: float,
    ) -> str:
        if score >= recommend_threshold:
            return "recommend"
        if score >= possible_threshold:
            return "possible"
        return "reject"


def load_schema_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_schema(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_matcher_from_args(args: argparse.Namespace) -> HybridSchemaMatcher:
    schema_records = load_schema_records(args.db)
    llm_client = None
    if args.api_base and args.model:
        llm_client = LocalOpenAICompatibleLLM(
            api_base=args.api_base,
            model=args.model,
            api_key=args.api_key,
            timeout_seconds=args.timeout,
        )
    return HybridSchemaMatcher(schema_records=schema_records, llm_client=llm_client)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hybrid JSON schema matcher: exact match -> vector recall -> LLM rerank."
    )
    parser.add_argument("--db", required=True, help="Path to the schema database JSON file.")
    parser.add_argument("--target", required=True, help="Path to the target schema JSON file.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of vector candidates to recall before reranking.",
    )
    parser.add_argument(
        "--api-base",
        default=os.getenv("LLM_SCHEMA_API_BASE", "http://127.0.0.1:1234/v1"),
        help="OpenAI-compatible API base URL, e.g. LM Studio local server.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("LLM_SCHEMA_MODEL", "qwen35-9b"),
        help="Model name exposed by the OpenAI-compatible API.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY", "lm-studio"),
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
        help="Skip stage 3 LLM reranking and use deterministic fallback only.",
    )

    args = parser.parse_args()
    if args.disable_llm:
        args.api_base = ""
        args.model = ""

    matcher = build_matcher_from_args(args)
    result = matcher.match_schema(load_schema(args.target), top_k=args.top_k)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
