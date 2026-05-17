-- ============================================================================
-- Migration: drop unused workflow surface + clean up CKN ingest contract
-- Date: 2026-05-16
--
-- Changes:
-- 1. DROP tables: support_tickets, scraper_jobs, automated_ingestion_artifacts,
--    submission_queue
-- 2. DROP ckn_auto_created columns (users, edge_devices, model_cards, models)
--
-- The ingest trigger function is defined in db/bootstrap_schema.sql (CREATE OR
-- REPLACE FUNCTION is idempotent); this migration only handles drops.
--
-- Idempotent. Safe to re-run.
-- ============================================================================

BEGIN;

-- 1. Drop deprecated workflow tables
DROP TABLE IF EXISTS automated_ingestion_artifacts CASCADE;
DROP TABLE IF EXISTS scraper_jobs CASCADE;
DROP TABLE IF EXISTS support_tickets CASCADE;
DROP TABLE IF EXISTS submission_queue CASCADE;

-- 2. Drop ckn_auto_created provenance flags
ALTER TABLE users        DROP COLUMN IF EXISTS ckn_auto_created;
ALTER TABLE edge_devices DROP COLUMN IF EXISTS ckn_auto_created;
ALTER TABLE model_cards  DROP COLUMN IF EXISTS ckn_auto_created;
ALTER TABLE models       DROP COLUMN IF EXISTS ckn_auto_created;

COMMIT;
