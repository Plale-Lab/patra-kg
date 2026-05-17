#!/usr/bin/env python3
"""Smoke test: verify LiteLLM on Tapis is reachable and responds to a simple prompt.

Fetches a fresh Tapis token automatically before each run.
Reads credentials from poc/.env via python-dotenv.

Usage:
    cd patra-knowledge-base
    python poc/smoke_test_litellm.py
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import httpx

load_dotenv(Path(__file__).parent.parent / ".env")

LITELLM_BASE = "https://litellm.pods.tacc.tapis.io"
TAPIS_AUTH_URL = "https://tacc.tapis.io/v3/oauth2/tokens"


def fetch_tapis_token() -> str:
    username = os.getenv("TAPIS_USERNAME", "").strip()
    password = os.getenv("TAPIS_PASSWORD", "").strip()
    if not username or not password:
        print("ERROR: Set TAPIS_USERNAME and TAPIS_PASSWORD env vars.")
        sys.exit(1)

    print(f"0. Fetching Tapis token for {username}...")
    try:
        resp = httpx.post(
            TAPIS_AUTH_URL,
            json={"username": username, "password": password, "grant_type": "password"},
            timeout=15,
        )
        resp.raise_for_status()
        token = resp.json()["result"]["access_token"]["access_token"]
        print(f"   OK — Token: {token[:20]}...")
        return token
    except httpx.HTTPStatusError as e:
        print(f"   FAIL — HTTP {e.response.status_code}: {e.response.text[:200]}")
        sys.exit(1)
    except (KeyError, TypeError) as e:
        print(f"   FAIL — Unexpected response format: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"   FAIL — {e}")
        sys.exit(1)


def main():
    token = fetch_tapis_token()
    headers = {"X-Tapis-Token": token}

    # --- 1. List models ---
    print(f"1. GET {LITELLM_BASE}/models")
    try:
        resp = httpx.get(f"{LITELLM_BASE}/models", headers=headers, timeout=15)
        resp.raise_for_status()
        models = [m["id"] for m in resp.json().get("data", []) if m.get("id")]
        print(f"   OK — {len(models)} model(s): {', '.join(models[:5])}")
        if not models:
            print("   WARN: No models returned. Endpoint may be misconfigured.")
            sys.exit(1)
    except httpx.HTTPStatusError as e:
        print(f"   FAIL — HTTP {e.response.status_code}: {e.response.text[:200]}")
        sys.exit(1)
    except Exception as e:
        print(f"   FAIL — {e}")
        sys.exit(1)

    # --- 2. Chat completion ---
    model = next((m for m in models if "llama" in m.lower()), models[0])
    print(f"\n2. POST {LITELLM_BASE}/v1/chat/completions  (model={model})")
    try:
        resp = httpx.post(
            f"{LITELLM_BASE}/v1/chat/completions",
            headers={**headers, "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Reply with exactly: PATRA_OK"}],
                "temperature": 0.0,
                "max_tokens": 20,
            },
            timeout=60,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        print(f"   OK — Response: {content}")
    except httpx.HTTPStatusError as e:
        print(f"   FAIL — HTTP {e.response.status_code}: {e.response.text[:200]}")
        sys.exit(1)
    except Exception as e:
        print(f"   FAIL — {e}")
        sys.exit(1)

    # --- 3. JSON structured output ---
    print(f"\n3. Structured JSON output test (model={model})")
    try:
        resp = httpx.post(
            f"{LITELLM_BASE}/v1/chat/completions",
            headers={**headers, "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "You are a metadata assistant. Given this model card:\n"
                            '{"name": "ResNet-50", "category": "Image Classification"}\n\n'
                            "Fill in the missing keywords field. Return ONLY valid JSON:\n"
                            '{"keywords": "<comma-separated keywords>", "confidence": 0.9}'
                        ),
                    }
                ],
                "temperature": 0.2,
                "max_tokens": 200,
            },
            timeout=60,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        parsed = json.loads(content)
        print(f"   OK — Parsed JSON: {json.dumps(parsed, indent=2)}")
    except json.JSONDecodeError:
        print(f"   WARN — Got response but not valid JSON: {content[:200]}")
    except httpx.HTTPStatusError as e:
        print(f"   FAIL — HTTP {e.response.status_code}: {e.response.text[:200]}")
        sys.exit(1)
    except Exception as e:
        print(f"   FAIL — {e}")
        sys.exit(1)

    print("\n--- All checks passed ---")


if __name__ == "__main__":
    main()
