# PATRA Backend Deployment Topology

## Canonical backend repository

Use `patra-knowledge-base` / `patra-backend` as the only long-term backend codebase. Deploy two pods from the same image:

- `patrabackend`: stable backend
- `patrabackend-dev`: development backend

Both pods can point at the same PostgreSQL database.

## Route gating

The backend now gates dev-only routes by environment flags:

- `ENABLE_ASK_PATRA`
- `ENABLE_AUTOMATED_INGESTION`
- `ENABLE_DOMAIN_EXPERIMENTS`

Stable backend should leave these off. Dev backend should enable them.

## Stable backend env

```json
{
  "DATABASE_URL": "<shared-db-url>",
  "JWT_SECRET": "<secret>",
  "ENABLE_ASK_PATRA": "false",
  "ENABLE_AUTOMATED_INGESTION": "false",
  "ENABLE_DOMAIN_EXPERIMENTS": "false"
}
```

## Dev backend env

```json
{
  "DATABASE_URL": "<shared-db-url>",
  "JWT_SECRET": "<secret>",
  "ENABLE_ASK_PATRA": "true",
  "ENABLE_AUTOMATED_INGESTION": "true",
  "ENABLE_DOMAIN_EXPERIMENTS": "true",
  "ASK_PATRA_LLM_API_BASE": "https://litellm.pods.tacc.tapis.io",
  "ASK_PATRA_LLM_MODEL": "llama3.3-70b-instruct",
  "ASK_PATRA_TAPIS_TOKEN": "<optional service token>"
}
```

## Shared tables

These belong to the shared product surface and can be read/written by both stable and dev backends:

- `model_cards`
- `datasheets`
- `submission_queue`
- `tickets`
- `ticket_comments`
- `users` / actor profile tables
- shared audit / changelog / version lineage tables

## Dev-only tables

These support experimental workflows and should only be written by `patrabackend-dev`:

- `scraper_jobs`
- `automated_ingestion_artifacts`
- `camera_trap_events`
- `camera_trap_power`
- `digital_ag_events`
- `digital_ag_power`

Ask Patra currently stores conversation memory and prompt files on a mounted volume rather than in PostgreSQL.

## Database rules

- One shared PostgreSQL instance is acceptable.
- Schema migrations must be forward compatible.
- Dev-only workflows must write isolated tables, not shared catalog tables.
- Promotion from dev-only state into shared catalog tables must be explicit.
