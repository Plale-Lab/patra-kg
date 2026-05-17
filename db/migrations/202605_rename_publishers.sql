-- Migration: rename publishers -> datasheet_publishers for prefix uniformity
-- with the rest of the DataCite child tables (datasheet_creators, datasheet_titles,
-- datasheet_subjects, etc.).
--
-- Note: publishers is a parent OF datasheets (M:1 shared), not a child like the
-- other datasheet_* tables — but the user prefers prefix uniformity over
-- naming-by-relationship.
--
-- Idempotent.

BEGIN;

DO $rename$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_class WHERE relname = 'publishers' AND relkind = 'r') THEN
    ALTER TABLE publishers RENAME TO datasheet_publishers;
  END IF;
END
$rename$;

COMMIT;
