import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from src.hybrid_schema_matcher import HybridSchemaMatcher, build_matcher_from_args
else:
    from .hybrid_schema_matcher import HybridSchemaMatcher, build_matcher_from_args


@dataclass
class ApiServerConfig:
    db: str
    api_base: str
    model: str
    api_key: str
    timeout: int
    disable_llm: bool
    top_k: int


@dataclass
class ApiServerState:
    matcher: HybridSchemaMatcher
    config: ApiServerConfig


def create_state(args: argparse.Namespace) -> ApiServerState:
    config = ApiServerConfig(
        db=args.db,
        api_base=args.api_base,
        model=args.model,
        api_key=args.api_key,
        timeout=args.timeout,
        disable_llm=args.disable_llm,
        top_k=args.top_k,
    )
    matcher = build_matcher_from_args(args)
    return ApiServerState(matcher=matcher, config=config)


def make_json_response(status: int, payload: Dict[str, Any]) -> Tuple[int, bytes]:
    body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    return status, body


def parse_request_json(raw_body: bytes) -> Dict[str, Any]:
    if not raw_body:
        raise ValueError("Request body is empty.")
    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError("JSON body must be an object.")
    return payload


def handle_match_request(state: ApiServerState, payload: Dict[str, Any]) -> Dict[str, Any]:
    target_schema = payload.get("target_schema")
    if not isinstance(target_schema, dict):
        raise ValueError("Field 'target_schema' must be a JSON object.")

    top_k = payload.get("top_k", state.config.top_k)
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("Field 'top_k' must be a positive integer.")

    result = state.matcher.match_schema(target_schema=target_schema, top_k=top_k)
    return asdict(result)


def build_handler(state: ApiServerState) -> type[BaseHTTPRequestHandler]:
    class SchemaMatchHTTPRequestHandler(BaseHTTPRequestHandler):
        server_version = "SchemaMatcherHTTP/0.1"

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._write_json(
                    HTTPStatus.OK,
                    {
                        "status": "ok",
                        "llm_enabled": not state.config.disable_llm,
                        "default_model": state.config.model,
                        "db": state.config.db,
                    },
                )
                return

            if self.path == "/config":
                self._write_json(
                    HTTPStatus.OK,
                    {
                        "db": state.config.db,
                        "api_base": state.config.api_base,
                        "model": state.config.model,
                        "timeout": state.config.timeout,
                        "top_k": state.config.top_k,
                        "disable_llm": state.config.disable_llm,
                    },
                )
                return

            self._write_json(
                HTTPStatus.NOT_FOUND,
                {"error": f"Unknown route: {self.path}"},
            )

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/match":
                self._write_json(
                    HTTPStatus.NOT_FOUND,
                    {"error": f"Unknown route: {self.path}"},
                )
                return

            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length)

            try:
                payload = parse_request_json(raw_body)
                result = handle_match_request(state, payload)
            except ValueError as exc:
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            except Exception as exc:  # noqa: BLE001
                self._write_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": "Schema matching failed.", "details": str(exc)},
                )
                return

            self._write_json(HTTPStatus.OK, result)

        def log_message(self, format: str, *args: object) -> None:
            return

        def _write_json(self, status: int, payload: Dict[str, Any]) -> None:
            _, body = make_json_response(status, payload)
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return SchemaMatchHTTPRequestHandler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the hybrid schema matcher as a small HTTP API.")
    parser.add_argument("--db", required=True, help="Path to the schema database JSON file.")
    parser.add_argument(
        "--host",
        default=os.getenv("LLM_SCHEMA_HOST", "127.0.0.1"),
        help="Host address to bind the API server to.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("LLM_SCHEMA_PORT", "8008")),
        help="Port to bind the API server to.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Default number of vector candidates to recall before reranking.",
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
        help="Use deterministic fallback only and skip LLM reranking.",
    )
    return parser


def run_server(args: argparse.Namespace) -> None:
    state = create_state(args)
    server = ThreadingHTTPServer((args.host, args.port), build_handler(state))
    print(
        json.dumps(
            {
                "status": "starting",
                "host": args.host,
                "port": args.port,
                "db": args.db,
                "model": args.model,
                "llm_enabled": not args.disable_llm,
            },
            ensure_ascii=False,
        )
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    parser = build_parser()
    run_server(parser.parse_args())


if __name__ == "__main__":
    main()
