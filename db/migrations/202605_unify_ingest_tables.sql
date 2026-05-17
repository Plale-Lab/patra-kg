-- Migration: unify CKN flat ingest tables into a single domain-aware schema.
-- Date: 2026-05-16
--
-- Changes:
-- 1. Drop digital_ag_events, digital_ag_power (empty, never used)
-- 2. Rename camera_trap_events -> events, camera_trap_power -> power_summary
-- 3. Add `domain text NOT NULL DEFAULT 'animal-ecology'` to both
--
-- Internal object names (indexes, triggers, trigger functions) keep their
-- original camera_trap_* prefixes — they're functional regardless of the
-- table they're attached to, and renaming them is purely cosmetic.
--
-- Idempotent. Safe to re-run.

BEGIN;

-- 1. Drop empty / unused per-domain duplicates
DROP TABLE IF EXISTS digital_ag_events CASCADE;
DROP TABLE IF EXISTS digital_ag_power CASCADE;

-- 2. Rename tables (only if old names still exist — idempotency guard)
DO $rename$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_class WHERE relname = 'camera_trap_events' AND relkind = 'r') THEN
    ALTER TABLE camera_trap_events RENAME TO events;
  END IF;
  IF EXISTS (SELECT 1 FROM pg_class WHERE relname = 'camera_trap_power' AND relkind = 'r') THEN
    ALTER TABLE camera_trap_power RENAME TO power_summary;
  END IF;
END
$rename$;

-- 3. Add domain column with default
ALTER TABLE events         ADD COLUMN IF NOT EXISTS domain text NOT NULL DEFAULT 'animal-ecology';
ALTER TABLE power_summary  ADD COLUMN IF NOT EXISTS domain text NOT NULL DEFAULT 'animal-ecology';

COMMIT;
