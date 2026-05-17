"""Constants shared across REST and MCP servers."""

# Valid domain values for the unified `events` / `power_summary` tables.
# Add new domain strings here when ingestion for that domain is wired up.
VALID_DOMAINS = frozenset({"animal-ecology"})
