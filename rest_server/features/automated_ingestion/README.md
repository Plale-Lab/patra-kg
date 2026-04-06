# Automated Ingestion

Dedicated ingestion pool for discovered CSV resources.

- Route entrypoint: `rest_server/routes/automated_ingestion.py`
- Models contract: `rest_server/ingestion_models.py`
- Stores staged CSV artifacts and review state separately from the main resource pool

