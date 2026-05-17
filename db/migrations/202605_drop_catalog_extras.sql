-- Migration: drop catalog-side tables that aren't needed
-- Date: 2026-05-16
--
-- Drops:
--   - asset_backups (asset versioning snapshots)
--   - asset_change_logs (audit trail)
--   - dataset_schemas (referenced by datasheets.dataset_schema_id; FK + column also dropped)
--   - generated_dataset_artifacts (agent_tools synthesized dataset persistence)
--
-- App-side cleanup landed in the same change set:
--   - rest_server/asset_backups.py deleted
--   - asset_change_logs INSERT/SELECT removed from routes
--   - dataset_schemas INSERT/JOIN/column removed from datasheets create/update + snapshot
--   - generated_dataset_artifacts INSERT/SELECT removed from agent_tools
--   - Pydantic models (AssetBackupRecord, AssetBackupRunResult, AssetChangeLogEntry,
--     AssetFieldChange) removed; AssetUpdateResult.backup_id removed
--
-- Idempotent. Safe to re-run.

BEGIN;

-- 1. Drop the FK column on datasheets first (so dataset_schemas can be dropped cleanly)
DROP INDEX IF EXISTS idx_datasheets_dataset_schema_id;
ALTER TABLE datasheets DROP COLUMN IF EXISTS dataset_schema_id;

-- 2. Drop the four tables (CASCADE handles any dangling FK constraints)
DROP TABLE IF EXISTS asset_backups CASCADE;
DROP TABLE IF EXISTS asset_change_logs CASCADE;
DROP TABLE IF EXISTS dataset_schemas CASCADE;
DROP TABLE IF EXISTS generated_dataset_artifacts CASCADE;

COMMIT;
