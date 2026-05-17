#!/usr/bin/env python3
"""Fetch 100 real HuggingFace model cards for augmentation evaluation.

50 top-downloaded (well-documented) + 50 sparse (<1000 downloads, diverse tasks).
Saves to poc/real_hf_cards.json + poc/real_input_cards/.

Usage:
    python poc/fetch_real_hf_cards.py
"""

import json
import time
from pathlib import Path

import httpx

HF_API = "https://huggingface.co"
OUTPUT_DIR = Path(__file__).parent / "data" / "inputs" / "real_input_cards"
OUTPUT_JSON = Path(__file__).parent / "data" / "inputs" / "real_hf_cards.json"

TASK_CATEGORIES = [
    "text-generation",
    "image-classification",
    "fill-mask",
    "text2text-generation",
    "automatic-speech-recognition",
    "object-detection",
    "token-classification",
    "text-to-image",
    "zero-shot-classification",
    "image-to-text",
    "audio-classification",
    "tabular-classification",
    "mask-generation",
    "image-text-to-text",
]


def fetch_models(params: dict, limit: int) -> list[dict]:
    url = f"{HF_API}/api/models"
    params = {**params, "limit": limit}
    resp = httpx.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def fetch_datasets(params: dict, limit: int) -> list[dict]:
    url = f"{HF_API}/api/datasets"
    params = {**params, "limit": limit}
    resp = httpx.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def fetch_readme(repo_id: str) -> str:
    url = f"{HF_API}/{repo_id}/resolve/main/README.md"
    for attempt in range(2):
        try:
            resp = httpx.get(url, timeout=10, follow_redirects=True)
            if resp.status_code == 200:
                text = resp.text
                if len(text) > 15000:
                    text = text[:15000]
                return text
            return ""
        except Exception:
            if attempt == 0:
                time.sleep(1)
    return ""


def collect_unique(models: list[dict], seen: set, target: int) -> list[dict]:
    result = []
    for m in models:
        if len(result) >= target:
            break
        repo_id = m.get("id") or m.get("modelId")
        if repo_id and repo_id not in seen:
            seen.add(repo_id)
            result.append(m)
    return result


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    seen_ids: set[str] = set()

    # --- Tier 1: 5 top-downloaded (well-documented) ---
    print("=== Tier 1: Top-downloaded models (5) ===")
    top_models = []
    filler = fetch_models({"sort": "downloads", "direction": "-1"}, limit=20)
    top_models = collect_unique(filler, seen_ids, target=5)
    for m in top_models:
        print(f"  {m.get('id'):<45s}  dl={m.get('downloads',0):>12,}")
    print(f"  Tier 1 total: {len(top_models)}")

    # --- Tier 2: 5 sparse models (<1000 downloads) ---
    print("\n=== Tier 2: Sparse models <1000 downloads (5) ===")
    sparse_models = []
    # Grab from diverse tasks
    for task in TASK_CATEGORIES[:10]:
        # Fetch recent models sorted by last modified (not downloads) — these tend to be smaller/newer
        models = fetch_models({"pipeline_tag": task, "sort": "lastModified", "direction": "-1"}, limit=20)
        # Filter to <1000 downloads
        low_dl = [m for m in models if (m.get("downloads") or 0) < 1000]
        added = collect_unique(low_dl, seen_ids, target=1)
        sparse_models.extend(added)
        if added:
            dl_range = f"{min(m.get('downloads', 0) for m in added)}-{max(m.get('downloads', 0) for m in added)}"
        else:
            dl_range = "none found"
        print(f"  {task:<35s}  +{len(added)}  downloads: {dl_range}")
        time.sleep(0.15)
        if len(sparse_models) >= 5:
            break

    if len(sparse_models) < 5:
        filler = fetch_models({"sort": "lastModified", "direction": "-1"}, limit=50)
        low_dl = [m for m in filler if (m.get("downloads") or 0) < 1000]
        extra = collect_unique(low_dl, seen_ids, target=5 - len(sparse_models))
        sparse_models.extend(extra)
        print(f"  (filled from recent low-download)  +{len(extra)}")

    sparse_models = sparse_models[:5]
    print(f"  Tier 2 total: {len(sparse_models)}")

    # --- Datasets: 5 popular + 5 sparse ---
    DATASET_TASKS = ["text-classification", "image-classification", "question-answering",
                     "automatic-speech-recognition", "text-generation", "object-detection",
                     "summarization", "translation", "token-classification", "tabular-classification"]

    print("\n=== Tier 3: Top-downloaded datasets (5) ===")
    top_datasets = []
    filler = fetch_datasets({"sort": "downloads", "direction": "-1"}, limit=20)
    top_datasets = collect_unique(filler, seen_ids, target=5)
    for d in top_datasets:
        print(f"  {d.get('id'):<45s}  dl={d.get('downloads',0):>12,}")
    print(f"  Tier 3 total: {len(top_datasets)}")

    print("\n=== Tier 4: Sparse datasets <1000 downloads (5) ===")
    sparse_datasets = []
    for task in DATASET_TASKS:
        ds_list = fetch_datasets({"sort": "lastModified", "direction": "-1"}, limit=20)
        low_dl = [d for d in ds_list if (d.get("downloads") or 0) < 1000]
        added = collect_unique(low_dl, seen_ids, target=1)
        sparse_datasets.extend(added)
        if added:
            dl_range = f"{min(d.get('downloads', 0) for d in added)}-{max(d.get('downloads', 0) for d in added)}"
        else:
            dl_range = "none found"
        print(f"  {task:<35s}  +{len(added)}  downloads: {dl_range}")
        time.sleep(0.15)
        if len(sparse_datasets) >= 5:
            break

    if len(sparse_datasets) < 5:
        filler = fetch_datasets({"sort": "lastModified", "direction": "-1"}, limit=50)
        low_dl = [d for d in filler if (d.get("downloads") or 0) < 1000]
        extra = collect_unique(low_dl, seen_ids, target=5 - len(sparse_datasets))
        sparse_datasets.extend(extra)
        print(f"  (filled from recent low-download)  +{len(extra)}")

    sparse_datasets = sparse_datasets[:5]
    print(f"  Tier 4 total: {len(sparse_datasets)}")

    # --- Combine all ---
    all_models = top_models + sparse_models
    all_datasets = top_datasets + sparse_datasets
    total = len(all_models) + len(all_datasets)
    print(f"\n=== Fetching READMEs for {total} entries ({len(all_models)} models + {len(all_datasets)} datasets) ===")

    dataset = []
    idx = 0

    # Models
    n_top_models = len(top_models)
    for i, model in enumerate(all_models):
        repo_id = model.get("id") or model.get("modelId")
        readme = fetch_readme(repo_id)
        tier = "top" if i < n_top_models else "sparse"
        idx += 1
        dataset.append({
            "id": idx, "asset_type": "model_card", "repo_id": repo_id,
            "domain": model.get("pipeline_tag") or "unknown",
            "tier": tier, "downloads": model.get("downloads", 0),
            "hf_api_response": model, "readme_body": readme, "patra_ground_truth": {},
        })
        slug = repo_id.replace("/", "_").lower()[:80]
        with open(OUTPUT_DIR / f"{idx:03d}_mc_{slug}.json", "w") as f:
            json.dump(model, f, indent=2)
        if idx % 5 == 0:
            print(f"  {idx}/{total} fetched")
        time.sleep(0.1)

    # Datasets
    n_top_ds = len(top_datasets)
    for i, ds in enumerate(all_datasets):
        repo_id = ds.get("id") or ""
        readme = fetch_readme(repo_id)
        tier = "top" if i < n_top_ds else "sparse"
        idx += 1
        dataset.append({
            "id": idx, "asset_type": "datasheet", "repo_id": repo_id,
            "domain": (ds.get("tags") or ["unknown"])[0] if ds.get("tags") else "unknown",
            "tier": tier, "downloads": ds.get("downloads", 0),
            "hf_api_response": ds, "readme_body": readme, "patra_ground_truth": {},
        })
        slug = repo_id.replace("/", "_").lower()[:80]
        with open(OUTPUT_DIR / f"{idx:03d}_ds_{slug}.json", "w") as f:
            json.dump(ds, f, indent=2)
        if idx % 5 == 0:
            print(f"  {idx}/{total} fetched")
        time.sleep(0.1)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(dataset, f, indent=2)

    # --- Stats ---
    mc_entries = [e for e in dataset if e["asset_type"] == "model_card"]
    ds_entries = [e for e in dataset if e["asset_type"] == "datasheet"]
    top_entries = [e for e in dataset if e["tier"] == "top"]
    sparse_entries = [e for e in dataset if e["tier"] == "sparse"]

    def _stats(entries, label):
        n = len(entries)
        if n == 0:
            return
        has_readme = sum(1 for e in entries if len(e["readme_body"]) > 100)
        has_tags_gt5 = sum(1 for e in entries if len(e["hf_api_response"].get("tags") or []) > 5)
        avg_dl = sum(e["downloads"] for e in entries) / max(n, 1)
        print(f"\n  {label} ({n}):")
        print(f"    README >100c:  {has_readme}/{n}")
        print(f"    tags >5:       {has_tags_gt5}/{n}")
        print(f"    avg downloads: {avg_dl:,.0f}")

    print(f"\nSaved {len(dataset)} entries to {OUTPUT_JSON}")
    print(f"  Model cards: {len(mc_entries)}, Datasheets: {len(ds_entries)}")
    _stats(mc_entries, "Model Cards")
    _stats(ds_entries, "Datasheets")
    _stats(top_entries, "Top-downloaded (all types)")
    _stats(sparse_entries, "Sparse (all types)")


if __name__ == "__main__":
    main()
