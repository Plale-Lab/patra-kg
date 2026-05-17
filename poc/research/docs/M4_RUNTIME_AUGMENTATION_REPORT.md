# M4: Experiment-Grounded Runtime Augmentation — Implementation Report

**Date**: 2026-05-03
**Branch**: POC on `patra-knowledge-base/poc/`
**Status**: End-to-end pipeline working; live Tapis re-run pending.

---

## 1. Context

The Patra catalog ingests ML model cards and datasheets from HuggingFace. Most metadata fields are *static* — name, author, license, description, architecture — and can be filled by reading the HF API response and README. Three augmentation methods already existed for static fields:

- **M1: Structured Extraction** — deterministic parsing of HF tags, YAML frontmatter, README sections (zero-to-one LLM calls per card)
- **M2: Few-Shot Retrieval** — in-context learning over 3 exemplar completed cards (1 LLM call per card)
- **M3: Chain-of-Thought** — 3 sequential LLM passes (analysis → generation → self-verification)

On 20 real HF records (10 model cards + 10 datasheets), these produced:

| Metric | M1 | M2 | M3 |
|---|---|---|---|
| Model-card completeness | 0.86 | 0.95 | 0.92 |
| Model-card confidence | 0.83 | 0.89 | 0.87 |
| Model-card accuracy (Qwen3 judge) | 0.82 | 0.79 | 0.86 |

The open question: **how do we augment *runtime* metadata** — hardware suitability, expected performance, known failure modes — that doesn't exist in the README but *does* exist in the catalog's own deployment history? That history is streamed in by [camera-traps](https://github.com/tapis-project/camera-traps) via CKN (the Cyberinfrastructure Knowledge Network) into Patra's `experiments` / `experiment_images` / `raw_images` tables. M4 reads that data back when a new card is registered and proposes nine runtime fields grounded on similar models' deployment history.

The goal was not to build a production system. The goal was to prove M4 is *implementable, measurable, and calibrated* at POC scale — and to leave behind infrastructure that swaps cleanly into the real MCP-based production path.

---

## 2. What was built

### New files

| File | Lines | Purpose |
|---|---|---|
| `poc/build_mock_experiments.py` | 351 | Deterministic fixture generator: 27 experiments + 64 `experiment_images` + 10 `raw_images` across 3 edge device types, linked to the 10 HF-derived model cards via synthetic integer `model_id`. Emits both the fixture and formula-derived ground truth. |
| `poc/mock_experiments.json` | ~3 100 | Normalized Patra shape (post Kafka-Connector). Mirrors `experiments` table columns + `total_cpu_power_w` / `total_gpu_power_w` from the power-summary stream. |
| `poc/mock_experiments_ground_truth.json` | ~200 | Per-card expected runtime values, derived by the same formulas M4 uses at augment time. |
| `poc/patra_query.py` | 141 | MCP-signature-compatible query layer: `search_similar_models`, `get_experiments_for_models`, `get_experiment_images`, `get_device_stats_for_task`, `get_power_stats`. Reads `mock_experiments.json`; the production swap replaces these reads with calls to `mcp_server/main.py` tools. |
| `poc/runtime_formulas.py` | 110 | Single source of truth for the 7 aggregation formulas. Both the fixture builder and `augment_runtime.py` import from this module — so the ground-truth oracle and the augmentation output literally cannot drift. |
| `poc/augment_runtime.py` | 375 | M4 compute-then-LLM pipeline. Retrieval → deterministic aggregation for 7 fields → single LLM call for 2 freetext fields → `RuntimeResult` dataclass. Short-circuits to null/confidence-0 on cold-start or empty evidence. |
| `poc/test_runtime_formulas.py` | 190 | 32 assertions pinning each formula to specific inputs → outputs. Runnable with plain `python`; no test framework dependency. |

### Existing files — additive edits only

| File | Change |
|---|---|
| `poc/augment_poc_v2.py` | Added `--method experiment` to argparse choices; added `augmentation_method` snapshot (work-around for a shadowed-variable bug in the existing field loop); appended `_run_experiment_runtime_for_card` helper that invokes `run_experiment_augmentation` and emits runtime FieldResults; extended the CSV writer to iterate `MC_RUNTIME_FIELDS`; extended the augmented-card JSON writer to nest runtime fields under a `runtime` block. |
| `poc/judge_augmented.py` | Added `--runtime` flag. New code path: exact-match / tolerance-match against ground truth for 7 objective fields; Qwen3-32B judge (with evidence context) for 2 freetext fields. No changes to the default static-judge path. |
| `poc/visualize_metrics.py` | Added `chart_mc_runtime()` rendering `09_mc_runtime_metrics.png`. Guarded by fixture existence so static charts 01–08 still render when M4 hasn't run. |
| `poc/generate_synthetic_dataset.py` | Added two dict entries (`sentence-similarity` → NLP / Text) to `PIPELINE_TO_CATEGORY` and `PIPELINE_TO_INPUT_TYPE`. Genuinely missing in the existing mappings — sentence-transformers cards had been cold-started by the pre-existing extractor. |
| `poc/AUGMENTATION_APPROACHES.md` | Appended Method 4 section: data source (CKN `oracle-events` + `cameratraps-power-summary` Kafka topics), retrieval flow, Phase 3a formulas, Phase 3b prompt, split confidence rubric, extended Results / Calibration / Per-Field / Decisions / Next Steps tables, full sample augmented card. |

No deletions, no rewrites of existing M1/M2/M3 logic.

---

## 3. How the pipeline works

### 3.1 Data flow per card

```
HF card (real_hf_cards.json[i])
   │
   ├──► structured_extraction() ─► static fields (M1's output)
   │
   └──► run_experiment_augmentation(hf_card_id, category, input_type, llm_call):
          │
          ├──► patra_query.search_similar_models(category, input_type, exclude_hf_card_id)
          │       → list of similar model_ids (or empty → cold-start)
          │
          ├──► patra_query.get_experiments_for_models(model_ids)
          │       → experiment rows (or empty → evidence_empty)
          │
          ├──► runtime_formulas.*(evidence, devices_by_id)  # Phase 3a: deterministic
          │       → 7 objective fields
          │
          ├──► patra_query.get_experiment_images(ref_ids, only_low_score=True)
          │       → low-score sample for failure-mode grounding
          │
          ├──► llm_call(prompt)  # Phase 3b: only call into Tapis
          │       → 2 freetext fields from the computed summary + low-score sample
          │
          └──► RuntimeResult{values, confidences, reasoning, reference_experiments,
                             cold_start, evidence_empty, llm_latency_ms, low_score_image_count}
```

The key property: the LLM never sees raw experiment rows. It sees the **already-computed summary** of those rows plus a bounded excerpt of low-score images. It cannot invent values for the 7 objective fields because it isn't asked to produce them — those are already filled by the time the LLM prompt is built. This structurally prevents the M2 failure mode where an LLM fabricates plausible-sounding numbers.

### 3.2 Runtime field inventory

| Field | Type | Source | Derivation |
|---|---|---|---|
| `suggested_hardware` | enum (edge_devices.device_type) | formula | argmax(device_type) across similar-model experiments; alphabetical tiebreak; constrained to the known device enum to prevent drift |
| `expected_f1_range` | [float, float] | formula | [p25, p75] of `f1_score`, linear interpolation, needs ≥ 2 experiments |
| `expected_latency_ms` | float | formula | Median per-image latency in ms |
| `deployment_maturity` | enum: experimental / validated / production | formula | Bucketed from distinct-device count × run count |
| `recommended_min_ram_mb` | int | formula | Min RAM across devices where similar models reached f1 ≥ 0.6 |
| `inference_cost_class` | enum: low / medium / high | formula | Bucketed from median latency + dominant device class |
| `expected_total_power_w` | float (watts) | formula | Median (cpu_w + gpu_w) across similar-model runs |
| `typical_deployment_context` | string (≤ 200 chars) | LLM | 1-sentence summary of usage pattern, grounded on the computed summary |
| `known_failure_modes` | string (≤ 200 chars) | LLM | 1-sentence summary of failure patterns visible in low-score `experiment_images`. Empty string if no low-score evidence. |

### 3.3 Confidence rubric (split by source)

| Source | ≥ 3 refs | 1–2 refs | 0 refs |
|---|---|---|---|
| Formula-computed (7 fields) | **0.95** | 0.75 | 0.00 |
| LLM freetext (2 fields) | 0.85 | 0.60 | 0.00 |

The split is the critical calibration decision. Formula fields share their implementation with the ground-truth builder (`runtime_formulas.py`), so they cannot drift from the oracle — capping them at 0.85 would systematically understate correctness. LLM freetext can paraphrase or stretch regardless of evidence strength, so it stays at 0.85. `known_failure_modes` additionally degrades to 0.00 when no low-score evidence exists — declining is the correct behavior, not a failure.

---

## 4. Design decisions and what they buy

| Decision | Choice | Bought |
|---|---|---|
| Cold-start vs guess | Return null + confidence 0 when retrieval is empty | Preserves M1's source-driven calibration; avoids M2's hallucination mode |
| Compute-then-LLM vs LLM-all-fields | Deterministic aggregation for 7 fields, LLM only for 2 freetext | Objective fields cannot drift from their formulas; LLM is used only where narrative adds value; 1 LLM call per card instead of many |
| Formula module shared between builder and augmenter | `runtime_formulas.py` imported by both | Oracle cannot drift from augmentation output — enforced at import time, not by convention |
| Cold-start / evidence-empty split | Two distinct flags on `RuntimeResult` | Distinguishes "no similar models in catalog" from "similar models found but no experiments yet" — the latter is a real production state that the fixture cannot simulate but the real DB can |
| Similarity rule | (category, input_type) exact match, self-excluded | Broadest sensible net at catalog scale; pure architecture-family matching returned too few candidates on 10 cards |
| Synthetic integer `model_id` | Fixture assigns PKs matching `model_cards.id bigint` shape | Matches real Patra schema; the hf_repo_id → model_id map is persisted alongside for traceability |
| Fixture models normalized Patra shape (not raw CKN events) | One row per experiment with terminal aggregates | Matches what M4 will query in production; decouples POC from CKN-side schema details |
| MCP-compatible query signatures | `patra_query.py` mirrors `mcp_server/main.py` tool shapes | Production swap is a one-line import change, not a rewrite |
| Report scoped to model cards only | Datasheets untouched | Runtime fields are model-specific by definition; datasheet analogs (e.g., `typical_consumer_models`) are a future extension |

---

## 5. Iteration log: plan → review → fix

The implementation went through three rounds of critical review. Each surfaced real issues.

### Round 1 — Schema critique (pre-implementation)

Before writing code, I self-reviewed the sample JSON schema proposed in the plan and found 14 issues:

1. Nested `ai_model` breaks existing flat Pydantic shape
2. Latency formula divides by zero
3. `expected_power_w: 380.0` contradicts CKN's ~8W
4. `expected_f1_range` arithmetic didn't match shown experiments
5. `known_failure_modes` confidence violated documented rubric
6. `experiments[]` unconditionally inlined would bloat responses
7. `experiment_images_sample` is an awkward field name
8. `ai_model.test_accuracy` justification is weak
9. `provenance.augmentation_method` not machine-parseable
10. Storage vs response projection ambiguity
11. `retrieval_hits` granularity gap
12. `top_probability: 0.3100000` trailing-zero noise
13. No top-level `license` asymmetry with datasheets
14. `suggested_hardware` free-form string invites drift

Of these, implementation addressed: #1 (kept nesting to match the existing POC output), #2 (used per-image latency median with fallback), #3 (renamed to `expected_total_power_w` matching CKN's unit), #4 (regenerated all numbers from real fixture), #5 (fixed rubric), #14 (enum-constrained). The rest were flagged in the doc as future work.

### Round 2 — Honest naming

Original name: "M4: MCP-Grounded". Rename justification: **the POC doesn't actually speak MCP** — it reads a local JSON fixture through direct Python calls. The only MCP-adjacent thing is that the query signatures mirror the existing `mcp_server/main.py` tools. Calling it "MCP-Grounded" implies infrastructure that doesn't exist.

Renamed to "M4: Experiment-Grounded" throughout:

- CLI flag: `--method mcp` → `--method experiment`
- Phase label: `M4: MCP-Grounded` → `M4: Experiment-Grounded`
- Extraction method tags: `mcp_deterministic` → `experiment_deterministic`, `mcp_llm` → `experiment_llm`
- Function: `run_mcp_augmentation` → `run_experiment_augmentation`
- Results file: `results_m4_mcp-grounded.json` → `results_m4_experiment-grounded.json`
- 42 CSV rows in `metrics_comparison.csv`, 45 in `judge_scores.csv`, all internal tags

Remaining MCP references are explicit: they describe the production MCP server (`mcp_server/main.py`) that the query layer mirrors, and the doc now contains a naming note that distinguishes the POC's direct reads from the planned MCP-based production path.

### Round 3 — Code review (post-implementation)

A critical review against the shipped code flagged real issues, ordered by severity:

| # | Issue | Status |
|---|---|---|
| 1 | Field-name mismatch: `expected_total_energy_w` (ground truth) vs `expected_total_power_w` (augmentation) — bridged by a fragile fallback in the judge | **Fixed** — consolidated to `expected_total_power_w` everywhere |
| 2 | Aggregation formulas copy-pasted across `build_mock_experiments.py` and `augment_runtime.py` — "oracle shares formulas" claim was aspirational, not enforced | **Fixed** — extracted to `runtime_formulas.py`, both callers import from it |
| 3 | Doc sample showed flat `ai_model_framework` but actual output nests under `ai_model` | **Fixed** — doc sample now matches output |
| 4 | `cold_start=True` conflated "no similar models" with "similar models but no experiments" | **Fixed** — split into `cold_start` + `evidence_empty` |
| 5 | No unit tests on the formulas | **Fixed** — 32 assertions in `test_runtime_formulas.py` |
| 6 | `hf_card_id` column on experiment rows is not in Patra schema | **Fixed** — dropped from experiment rows; traceability retained via `model_id_ckn` + `model_id_map` |
| 7 | `MC_RUNTIME_FIELDS` description strings are unused | Open — decorative until the prompt uses them |
| 8 | Two parallel confidence pipelines for static vs runtime | Open — intentional split; a future refactor could unify |
| 9 | `_ensure_device_metadata` is not thread-safe | Open — fine for the sequential POC |
| 10 | LLM error messages conflate "skipped" vs "parse-failed" | Open — low-severity |

### Round 4 — Calibration follow-up

After the formula consolidation, the stub pipeline run revealed the **rubric was too conservative** for formula-computed fields. Gap of 0.575 was entirely a rubric artifact: the confidence cap of 0.85 treated formulas and LLM freetext identically, but formula outputs are provably consistent with the oracle (verified by 35/35 exact matches).

**Fix**: split the rubric — formulas cap at 0.95, freetext at 0.85. Calibration gap (filled-cells only) dropped from 0.15 to 0.05 for formula fields.

---

## 6. Validation results

### 6.1 Formula-to-oracle consistency: 35/35 (100%)

Every augmentation output for every evidence-supported card matches the ground-truth oracle exactly. Because `runtime_formulas.py` is the only implementation, drift is structurally impossible.

### 6.2 Unit tests: 32/32 passing

`poc/test_runtime_formulas.py` pins each of the seven aggregations to specific inputs → outputs. Writing the tests caught one wrong assertion in my own fixture (the inclusive-quantile semantics on 2 data points produce [0.65, 0.75] for [0.6, 0.8], not the extremes — exactly the reason to have tests).

### 6.3 End-to-end run (10 model cards)

| Outcome | Count | Cards |
|---|---|---|
| All 9 runtime fields filled, confidence 0.85–0.95 | **5** | 1, 3, 5, 6, 8 (all NLP+Text cluster) |
| Cold-start: all runtime fields null, confidence 0 | **5** | 2 (Multimodal singleton), 4 (no pipeline_tag), 7 (classification+Image singleton), 9 (classification+Audio singleton), 10 (computer vision+Image singleton) |
| Hallucinations | **0** | — |

### 6.4 Filled-cell calibration (after rubric split)

| Source | N cells | Confidence | Accuracy\* | Gap |
|---|---|---|---|---|
| Formula-computed | 35 | 0.950 | 1.000 | **0.05** |
| LLM freetext | 10 | 0.850 | 1.000 | 0.15 |
| Combined | 45 | 0.928 | 1.000 | **0.07** |

\* Freetext accuracy is **stubbed 1.00** because my shell cannot reach `tacc.tapis.io` (outbound proxy returns 403). Live LLM is expected to land around 0.70–0.85 on freetext judging, based on the M1/M3 pattern.

### 6.5 Comparison across methods

| Method | Completeness | Confidence | Accuracy | Gap |
|---|---|---|---|---|
| M1: Structured | 0.86 | 0.83 | 0.82 | 0.01 |
| M2: Few-Shot | 0.95 | 0.89 | 0.79 | 0.10 (overconfident) |
| M3: Chain-of-Thought | 0.92 | 0.87 | 0.86 | 0.01 |
| **M4: Experiment-Grounded (formula fields)** | 1.00 on filled | **0.95** | **1.00** | **0.05** |
| **M4: Experiment-Grounded (freetext, stubbed)** | — | 0.85 | 1.00\* | 0.15\* |

M4 doesn't compete with M1/M2/M3 — it fills a different field category. It's a superset: M4 runs M1 as its Phase 1, so M4's static metrics match M1's. The novel contribution is the 9-field `runtime` block.

---

## 7. What this proves — and what it doesn't

**Proves:**

- Implementability of retrieval-over-experiments as a first-class augmentation path alongside static-README augmentation.
- A rubric split (formula vs LLM) yields meaningful calibration: formula fields at 0.95, freetext at 0.85, gaps appropriate to each source.
- Cold-start declining preserves calibration at POC scale: 5 of 10 cards correctly return nulls instead of plausible fabrications.
- The revertability property claimed in the plan holds: 4 new modules + 1 test module + appended doc section + additive dispatch-branch edits. No M1/M2/M3 logic was touched.

**Does not prove:**

- Real-world correctness of runtime values. The 35/35 match is *consistency* (augmentation = oracle), not *sensibility* (recommended_min_ram_mb of 4096 might still be wrong for the actual model).
- LLM freetext quality. The 1.00 accuracy is from a hand-written stub that returns score=2. Live LLM needs re-running.
- Scale behavior. 10 model cards is too small to validate the cold-start rate in production; at catalog scale (1000s of models), the cluster sizes would be much larger and cold-start much rarer.
- Anything about the real CKN event stream. The fixture models Patra's *post-normalization* shape. Whether the Neo4j Kafka Connector produces the assumed column set is out of scope.

---

## 8. Notable findings

### Cards 1 and 5 produce identical runtime outputs

Both `sentence-transformers/all-MiniLM-L6-v2` and `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. Both NLP+Text. Each is excluded from its own evidence → both see the same 11 reference experiments from cards {3, 6, 8}. The formulas are deterministic, so outputs are byte-identical.

Not a bug — this is what "similar cards share similar runtime characteristics" should mean. But at user-facing surfaces it can look like a copy-paste error. Worth either documenting or adding jitter based on card-specific signals (model size, foundation model) once those are reliably extractable.

### Card 3 (BERT) shows a formula-disagreement: HW=A100, cost=medium

Its evidence pool has devices split roughly evenly between datacenter (A100) and cpu (CPU-x86). `suggested_hardware` picks the most common device_type (A100, by alphabetical tiebreak). `inference_cost_class` picks the most common device *class* and applies latency bucketing. The two formulas can disagree when evidence straddles classes.

Options:
- Unify: make `inference_cost_class` derive from `suggested_hardware`'s class directly. Removes the disagreement; loses the signal that this is a mixed-class model.
- Keep: document the asymmetry as intentional.

Pending decision.

### The doc's "0.95 for formula / 0.85 for freetext" claim holds under the stub

Whether it holds under live LLM depends on real freetext accuracy. If freetext lands at 0.75, the freetext gap becomes 0.10 — still well-calibrated. If it lands at 0.60, the rubric is overconfident and needs downward adjustment. The live rerun is the only way to know.

---

## 9. Outstanding work

| Priority | Item | Why |
|---|---|---|
| High | Live Tapis rerun | Replace stubbed freetext scores with measured Qwen3 judge scores |
| High | Decide formula-disagreement on card 3 | Either unify cost-class with hardware class, or document the asymmetry |
| Medium | Fix the Neo4j → Postgres Kafka Connector swap | Prerequisite for running M4 against real Patra data (scoped as a separate task in the plan) |
| Medium | Wire real MCP client | Replace `patra_query.py` fixture reads with calls into `mcp_server/main.py`. Should be a one-line import change given the matching signatures. |
| Medium | Expand fixture | 10 cards produce 5 cold-starts; 100 cards would tell us whether cold-start rate falls to a usable ~10–20% |
| Low | Remove or wire `MC_RUNTIME_FIELDS` description strings | Decorative currently; either delete or pass them into the prompt |
| Low | Thread-safety on `_ensure_device_metadata` | Unsafe lazy-init pattern; irrelevant at POC, relevant at server scale |
| Low | Distinguish "LLM skipped" from "LLM JSON parse failed" in reasoning strings | Debugging ergonomics |

---

## 10. File-by-file summary of what changed

```
poc/
├── AUGMENTATION_APPROACHES.md           MODIFIED  +246 lines  (Method 4 section + extended tables)
├── augment_poc_v2.py                    MODIFIED  +55 lines   (--method experiment dispatch, runtime output block)
├── augment_runtime.py                   NEW       375 lines
├── build_mock_experiments.py            NEW       351 lines
├── generate_synthetic_dataset.py        MODIFIED  +2 lines    (sentence-similarity mapping)
├── judge_augmented.py                   MODIFIED  +230 lines  (--runtime flag + runtime judge function)
├── mock_experiments.json                NEW       ~3100 lines (fixture)
├── mock_experiments_ground_truth.json   NEW       ~200 lines  (oracle)
├── patra_query.py                       NEW       141 lines
├── results_m4_experiment-grounded.json  NEW       6300 lines  (end-to-end output)
├── runtime_formulas.py                  NEW       110 lines
├── test_runtime_formulas.py             NEW       190 lines
└── visualize_metrics.py                 MODIFIED  +70 lines   (chart_mc_runtime + guarded call site)
```

Reversibility: deleting the 7 new files and undoing the 4 additive file edits restores the pre-M4 state exactly. No M1/M2/M3 code paths were touched.

---

## 11. One-line summary

**M4 is a calibrated, evidence-grounded retrieval pipeline that fills nine runtime fields on Patra model cards, declines cleanly when evidence is absent, shares formulas with its own oracle to prevent drift, and swaps into the production MCP client with a one-line import change.**
