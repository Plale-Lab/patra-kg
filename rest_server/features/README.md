# Backend Feature Modules

This directory groups PATRA backend logic by feature instead of keeping all feature-specific code directly under `routes/`.

Current modules:
- `ask_patra`: conversational assistant service, prompts, and models
- `automated_ingestion`: isolated CSV ingestion pipeline
- `agent_toolkit`: schema-search-oriented AI tooling docs and support files
- `resource_records`: record editing/search domain docs
- `shared`: reusable OpenAI-compatible provider helpers

HTTP entrypoints still live under `rest_server/routes/` and import from these feature modules.
