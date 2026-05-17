#!/usr/bin/env python3
"""Quick test: verify qwen3-32b works on LiteLLM."""

import json, os, sys
from pathlib import Path
from dotenv import load_dotenv
import httpx

load_dotenv(Path(__file__).parent.parent / ".env")

LITELLM = "https://litellm.pods.tacc.tapis.io"

# Auth
resp = httpx.post("https://tacc.tapis.io/v3/oauth2/tokens",
    json={"username": os.getenv("TAPIS_USERNAME"), "password": os.getenv("TAPIS_PASSWORD"), "grant_type": "password"}, timeout=15)
token = resp.json()["result"]["access_token"]["access_token"]
headers = {"X-Tapis-Token": token, "Content-Type": "application/json"}

# List models
print("Available models:")
resp = httpx.get(f"{LITELLM}/models", headers=headers, timeout=15)
for m in resp.json().get("data", []):
    print(f"  {m['id']}")

# Test qwen3-32b
print("\nTesting qwen3-32b (with /no_think suffix)...")
resp = httpx.post(f"{LITELLM}/v1/chat/completions", headers=headers, json={
    "model": "qwen3-32b",
    "messages": [
        {"role": "system", "content": "You are a concise assistant. Do not use <think> tags. Respond directly."},
        {"role": "user", "content": "Reply with exactly: JUDGE_OK"},
    ],
    "temperature": 0.0, "max_tokens": 20,
    "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
}, timeout=60)
resp.raise_for_status()
raw = resp.json()["choices"][0]["message"]["content"].strip()
# Strip <think>...</think> if present
import re
content = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
print(f"  Raw: {raw[:100]}")
print(f"  Cleaned: {content}")

# Test structured judge output
print("\nTesting judge-style prompt...")
resp = httpx.post(f"{LITELLM}/v1/chat/completions", headers=headers, json={
    "model": "qwen3-32b",
    "messages": [
        {"role": "system", "content": "You are a metadata quality judge. Return ONLY valid JSON. No thinking, no explanation, no markdown fences."},
        {"role": "user", "content": """Rate this augmented metadata value.

Field: category
Value: "classification"
Source model: microsoft/resnet-50 (image classification model)

Score 0-2:
  0 = wrong or misleading
  1 = acceptable but imprecise
  2 = correct and useful

Return ONLY valid JSON: {"score": N, "reason": "one sentence"}"""},
    ],
    "temperature": 0.0, "max_tokens": 200,
    "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
}, timeout=60)
resp.raise_for_status()
raw = resp.json()["choices"][0]["message"]["content"].strip()
# Strip think tags and markdown fences
content = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
content = re.sub(r"^```(?:json)?\s*", "", content)
content = re.sub(r"\s*```$", "", content)
print(f"  Raw: {raw[:200]}")
print(f"  Cleaned: {content}")
try:
    parsed = json.loads(content)
    print(f"  Parsed: score={parsed['score']}, reason={parsed['reason']}")
except Exception as e:
    print(f"  Failed to parse: {e}")

# Also test llama4-17b for comparison
print("\nTesting llama4-17b (generator model)...")
resp = httpx.post(f"{LITELLM}/v1/chat/completions", headers=headers, json={
    "model": "llama4-17b",
    "messages": [{"role": "user", "content": """Rate this augmented metadata value.

Field: category
Value: "classification"
Source model: microsoft/resnet-50 (image classification model)

Score 0-2:
  0 = wrong or misleading
  1 = acceptable but imprecise
  2 = correct and useful

Return ONLY valid JSON: {"score": N, "reason": "one sentence"}"""}],
    "temperature": 0.0, "max_tokens": 100,
}, timeout=60)
resp.raise_for_status()
content = resp.json()["choices"][0]["message"]["content"].strip()
print(f"  Response: {content}")
try:
    parsed = json.loads(content)
    print(f"  Parsed: score={parsed['score']}, reason={parsed['reason']}")
except:
    print("  Failed to parse")
