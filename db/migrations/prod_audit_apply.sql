-- ============================================================================
-- Production migration: bring prod patradb to current local schema.
-- Single transactional script. Atomic: BEGIN..COMMIT, rolls back on any error.
-- Idempotent: safe to re-run if it failed partway.
--
-- Combines (in order):
--   1. 202605_drop_workflow_and_revamp_ckn.sql  (drop legacy workflow tables)
--   2. 202605_drop_catalog_extras.sql           (drop asset_backups etc)
--   3. 202605_rename_publishers.sql             (publishers → datasheet_publishers)
--   4. 202605_unify_ingest_tables.sql           (camera_trap_* → events/power_summary)
--   5. Schema audit                             (drop versioning, audit cols, FK retypes, …)
--
-- BEFORE RUNNING: take a backup. There is no automatic restore on COMMIT.
-- ============================================================================

BEGIN;

-- ============================================================================
-- PHASE 0: Pre-flight
-- ============================================================================
DO $preflight$
DECLARE v_n int;
BEGIN
  EXECUTE $$SELECT COUNT(*) FROM public.users WHERE username IS NULL OR username = ''$$ INTO v_n;
  IF v_n > 0 THEN
    RAISE EXCEPTION 'PREFLIGHT FAIL: % users rows have NULL/empty username', v_n;
  END IF;
  EXECUTE $$SELECT COUNT(*) FROM (SELECT username FROM public.users GROUP BY username HAVING COUNT(*)>1) x$$ INTO v_n;
  IF v_n > 0 THEN
    RAISE EXCEPTION 'PREFLIGHT FAIL: % duplicate usernames', v_n;
  END IF;

  EXECUTE $$SELECT COUNT(*) FROM public.edge_devices WHERE device_id IS NULL OR device_id = ''$$ INTO v_n;
  IF v_n > 0 THEN
    RAISE EXCEPTION 'PREFLIGHT FAIL: % edge_devices rows have NULL/empty device_id', v_n;
  END IF;
  EXECUTE $$SELECT COUNT(*) FROM (SELECT device_id FROM public.edge_devices GROUP BY device_id HAVING COUNT(*)>1) x$$ INTO v_n;
  IF v_n > 0 THEN
    RAISE EXCEPTION 'PREFLIGHT FAIL: % duplicate device_ids', v_n;
  END IF;

  IF (SELECT data_type FROM information_schema.columns
      WHERE table_schema='public' AND table_name='experiments' AND column_name='user_id') = 'bigint' THEN
    EXECUTE $$SELECT COUNT(*) FROM public.experiments e LEFT JOIN public.users u ON u.id = e.user_id WHERE u.id IS NULL$$ INTO v_n;
    IF v_n > 0 THEN
      RAISE EXCEPTION 'PREFLIGHT FAIL: % experiments rows have orphan user_id', v_n;
    END IF;
  END IF;
  IF (SELECT data_type FROM information_schema.columns
      WHERE table_schema='public' AND table_name='experiments' AND column_name='edge_device_id') = 'bigint' THEN
    EXECUTE $$SELECT COUNT(*) FROM public.experiments e LEFT JOIN public.edge_devices d ON d.id = e.edge_device_id WHERE d.id IS NULL$$ INTO v_n;
    IF v_n > 0 THEN
      RAISE EXCEPTION 'PREFLIGHT FAIL: % experiments rows have orphan edge_device_id', v_n;
    END IF;
  END IF;

  RAISE NOTICE 'OK pre-flight passed';
END $preflight$;

-- ============================================================================
-- PHASE 1: Drop legacy workflow tables (202605_drop_workflow_and_revamp_ckn)
-- ============================================================================
DROP TABLE IF EXISTS automated_ingestion_artifacts CASCADE;
DROP TABLE IF EXISTS scraper_jobs                 CASCADE;
DROP TABLE IF EXISTS support_tickets              CASCADE;
DROP TABLE IF EXISTS submission_queue             CASCADE;
ALTER TABLE users        DROP COLUMN IF EXISTS ckn_auto_created;
ALTER TABLE edge_devices DROP COLUMN IF EXISTS ckn_auto_created;
ALTER TABLE model_cards  DROP COLUMN IF EXISTS ckn_auto_created;
ALTER TABLE models       DROP COLUMN IF EXISTS ckn_auto_created;

-- ============================================================================
-- PHASE 2: Drop catalog extras (202605_drop_catalog_extras)
-- ============================================================================
DROP INDEX  IF EXISTS idx_datasheets_dataset_schema_id;
ALTER TABLE datasheets DROP COLUMN IF EXISTS dataset_schema_id;
DROP TABLE  IF EXISTS asset_backups               CASCADE;
DROP TABLE  IF EXISTS asset_change_logs           CASCADE;
DROP TABLE  IF EXISTS dataset_schemas             CASCADE;
DROP TABLE  IF EXISTS generated_dataset_artifacts CASCADE;

-- ============================================================================
-- PHASE 3: Rename publishers → datasheet_publishers
-- ============================================================================
DO $$ BEGIN
  IF EXISTS (SELECT 1 FROM pg_class WHERE relname='publishers' AND relkind='r')
     AND NOT EXISTS (SELECT 1 FROM pg_class WHERE relname='datasheet_publishers' AND relkind='r') THEN
    ALTER TABLE publishers RENAME TO datasheet_publishers;
  END IF;
END $$;

-- ============================================================================
-- PHASE 4: Unify ingest tables (camera_trap_* → events/power_summary, drop digital_ag_*)
-- ============================================================================
DROP TABLE IF EXISTS digital_ag_events CASCADE;
DROP TABLE IF EXISTS digital_ag_power  CASCADE;

DO $$ BEGIN
  IF EXISTS (SELECT 1 FROM pg_class WHERE relname='camera_trap_events' AND relkind='r')
     AND NOT EXISTS (SELECT 1 FROM pg_class WHERE relname='events' AND relkind='r') THEN
    ALTER TABLE camera_trap_events RENAME TO events;
  END IF;
  IF EXISTS (SELECT 1 FROM pg_class WHERE relname='camera_trap_power' AND relkind='r')
     AND NOT EXISTS (SELECT 1 FROM pg_class WHERE relname='power_summary' AND relkind='r') THEN
    ALTER TABLE camera_trap_power RENAME TO power_summary;
  END IF;
END $$;

DO $$ BEGIN
  IF EXISTS (SELECT 1 FROM pg_class WHERE relname='events' AND relkind='r') THEN
    ALTER TABLE events ADD COLUMN IF NOT EXISTS domain text NOT NULL DEFAULT 'animal-ecology';
  END IF;
  IF EXISTS (SELECT 1 FROM pg_class WHERE relname='power_summary' AND relkind='r') THEN
    ALTER TABLE power_summary ADD COLUMN IF NOT EXISTS domain text NOT NULL DEFAULT 'animal-ecology';
  END IF;
END $$;

-- ============================================================================
-- PHASE 5: model_cards / datasheets uuid column reconciliation
-- prod uses model_cards.patra_mc_id (rename) and has no datasheets.uuid (add)
-- ============================================================================
CREATE EXTENSION IF NOT EXISTS pgcrypto;

DO $$ BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.columns
             WHERE table_schema='public' AND table_name='model_cards' AND column_name='patra_mc_id')
     AND NOT EXISTS (SELECT 1 FROM information_schema.columns
                     WHERE table_schema='public' AND table_name='model_cards' AND column_name='uuid') THEN
    ALTER TABLE model_cards RENAME COLUMN patra_mc_id TO uuid;
  END IF;
END $$;

ALTER TABLE model_cards ADD COLUMN IF NOT EXISTS uuid uuid DEFAULT gen_random_uuid();
UPDATE model_cards SET uuid = gen_random_uuid() WHERE uuid IS NULL;
ALTER TABLE model_cards ALTER COLUMN uuid SET NOT NULL;
ALTER TABLE model_cards ALTER COLUMN uuid SET DEFAULT gen_random_uuid();

ALTER TABLE datasheets ADD COLUMN IF NOT EXISTS uuid uuid DEFAULT gen_random_uuid();
UPDATE datasheets SET uuid = gen_random_uuid() WHERE uuid IS NULL;
ALTER TABLE datasheets ALTER COLUMN uuid SET NOT NULL;
ALTER TABLE datasheets ALTER COLUMN uuid SET DEFAULT gen_random_uuid();

-- ============================================================================
-- PHASE 6: Drop versioning + status; ensure single-column UNIQUE(uuid)
-- ============================================================================
ALTER TABLE model_cards DROP CONSTRAINT IF EXISTS model_cards_uuid_version_key;
ALTER TABLE model_cards DROP COLUMN IF EXISTS asset_version;
ALTER TABLE model_cards DROP COLUMN IF EXISTS previous_version_id;
ALTER TABLE model_cards DROP COLUMN IF EXISTS root_version_id;
ALTER TABLE model_cards DROP COLUMN IF EXISTS status;
DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname='model_cards_uuid_key') THEN
    ALTER TABLE model_cards ADD CONSTRAINT model_cards_uuid_key UNIQUE (uuid);
  END IF;
END $$;

ALTER TABLE datasheets DROP CONSTRAINT IF EXISTS datasheets_uuid_version_key;
ALTER TABLE datasheets DROP COLUMN IF EXISTS asset_version;
ALTER TABLE datasheets DROP COLUMN IF EXISTS previous_version_id;
ALTER TABLE datasheets DROP COLUMN IF EXISTS root_version_id;
DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname='datasheets_uuid_key') THEN
    ALTER TABLE datasheets ADD CONSTRAINT datasheets_uuid_key UNIQUE (uuid);
  END IF;
END $$;

-- ============================================================================
-- PHASE 7: experiments — rename/add experiment_uid, drop legacy cols
-- ============================================================================
DO $$ BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.columns
             WHERE table_schema='public' AND table_name='experiments' AND column_name='ckn_experiment_id')
     AND NOT EXISTS (SELECT 1 FROM information_schema.columns
                     WHERE table_schema='public' AND table_name='experiments' AND column_name='experiment_uid') THEN
    ALTER TABLE experiments RENAME COLUMN ckn_experiment_id TO experiment_uid;
  END IF;
  IF EXISTS (SELECT 1 FROM pg_constraint WHERE conname='experiments_ckn_experiment_id_key')
     AND NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname='experiments_experiment_uid_key') THEN
    ALTER TABLE experiments RENAME CONSTRAINT experiments_ckn_experiment_id_key TO experiments_experiment_uid_key;
  END IF;
END $$;
ALTER TABLE experiments ADD COLUMN IF NOT EXISTS experiment_uid text;
DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname='experiments_experiment_uid_key') THEN
    ALTER TABLE experiments ADD CONSTRAINT experiments_experiment_uid_key UNIQUE (experiment_uid);
  END IF;
END $$;
ALTER TABLE experiments DROP COLUMN IF EXISTS map_75;
ALTER TABLE experiments DROP COLUMN IF EXISTS submitted_at;
ALTER TABLE experiments DROP COLUMN IF EXISTS model_used_at;
ALTER TABLE experiments DROP COLUMN IF EXISTS created_at;
ALTER TABLE experiments DROP COLUMN IF EXISTS updated_at;
ALTER TABLE experiments ADD COLUMN IF NOT EXISTS total_cpu_power_w numeric(10,4);
ALTER TABLE experiments ADD COLUMN IF NOT EXISTS total_gpu_power_w numeric(10,4);

-- ============================================================================
-- PHASE 8: raw_images — rename/add image_uid, drop audit cols
-- ============================================================================
DO $$ BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.columns
             WHERE table_schema='public' AND table_name='raw_images' AND column_name='ckn_uuid')
     AND NOT EXISTS (SELECT 1 FROM information_schema.columns
                     WHERE table_schema='public' AND table_name='raw_images' AND column_name='image_uid') THEN
    ALTER TABLE raw_images RENAME COLUMN ckn_uuid TO image_uid;
  END IF;
  IF EXISTS (SELECT 1 FROM pg_constraint WHERE conname='raw_images_ckn_uuid_key')
     AND NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname='raw_images_image_uid_key') THEN
    ALTER TABLE raw_images RENAME CONSTRAINT raw_images_ckn_uuid_key TO raw_images_image_uid_key;
  END IF;
END $$;
ALTER TABLE raw_images ADD COLUMN IF NOT EXISTS image_uid text;
DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname='raw_images_image_uid_key') THEN
    ALTER TABLE raw_images ADD CONSTRAINT raw_images_image_uid_key UNIQUE (image_uid);
  END IF;
END $$;
ALTER TABLE raw_images DROP COLUMN IF EXISTS created_at;
ALTER TABLE raw_images DROP COLUMN IF EXISTS updated_at;

-- ============================================================================
-- PHASE 9: experiments FK type change (bigint → text)
-- ============================================================================
DO $$ BEGIN
  IF (SELECT data_type FROM information_schema.columns
      WHERE table_schema='public' AND table_name='experiments' AND column_name='user_id') = 'bigint' THEN
    ALTER TABLE experiments ADD COLUMN user_id_new text;
    UPDATE experiments e SET user_id_new = u.username FROM users u WHERE e.user_id = u.id;
    IF EXISTS (SELECT 1 FROM experiments WHERE user_id_new IS NULL) THEN
      RAISE EXCEPTION 'BACKFILL FAIL: experiments.user_id had unresolvable rows';
    END IF;
    ALTER TABLE experiments DROP CONSTRAINT IF EXISTS experiments_user_id_fkey;
    ALTER TABLE experiments DROP COLUMN user_id;
    ALTER TABLE experiments RENAME COLUMN user_id_new TO user_id;
    ALTER TABLE experiments ALTER COLUMN user_id SET NOT NULL;
  END IF;
  IF (SELECT data_type FROM information_schema.columns
      WHERE table_schema='public' AND table_name='experiments' AND column_name='edge_device_id') = 'bigint' THEN
    ALTER TABLE experiments ADD COLUMN edge_device_id_new text;
    UPDATE experiments e SET edge_device_id_new = d.device_id FROM edge_devices d WHERE e.edge_device_id = d.id;
    IF EXISTS (SELECT 1 FROM experiments WHERE edge_device_id_new IS NULL) THEN
      RAISE EXCEPTION 'BACKFILL FAIL: experiments.edge_device_id had unresolvable rows';
    END IF;
    ALTER TABLE experiments DROP CONSTRAINT IF EXISTS experiments_edge_device_id_fkey;
    ALTER TABLE experiments DROP COLUMN edge_device_id;
    ALTER TABLE experiments RENAME COLUMN edge_device_id_new TO edge_device_id;
    ALTER TABLE experiments ALTER COLUMN edge_device_id SET NOT NULL;
  END IF;
END $$;

-- ============================================================================
-- PHASE 10: users rebuild — drop bigint id, promote username to PK
-- ============================================================================
DO $$ BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.columns
             WHERE table_schema='public' AND table_name='users' AND column_name='id') THEN
    ALTER TABLE users DROP CONSTRAINT IF EXISTS users_pkey;
    ALTER TABLE users DROP CONSTRAINT IF EXISTS users_ckn_user_id_key;
    ALTER TABLE users DROP CONSTRAINT IF EXISTS users_username_key;
    ALTER TABLE users DROP COLUMN id;
    ALTER TABLE users DROP COLUMN IF EXISTS ckn_user_id;
    ALTER TABLE users DROP COLUMN IF EXISTS created_at;
    ALTER TABLE users DROP COLUMN IF EXISTS updated_at;
    ALTER TABLE users ALTER COLUMN username SET NOT NULL;
    ALTER TABLE users ADD PRIMARY KEY (username);
  END IF;
END $$;

-- ============================================================================
-- PHASE 11: edge_devices rebuild — drop bigint id, promote device_id to PK; add geo
-- ============================================================================
DO $$ BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.columns
             WHERE table_schema='public' AND table_name='edge_devices' AND column_name='id') THEN
    ALTER TABLE edge_devices DROP CONSTRAINT IF EXISTS edge_devices_pkey;
    ALTER TABLE edge_devices DROP CONSTRAINT IF EXISTS edge_devices_device_id_key;
    ALTER TABLE edge_devices DROP COLUMN id;
    ALTER TABLE edge_devices DROP COLUMN IF EXISTS created_at;
    ALTER TABLE edge_devices DROP COLUMN IF EXISTS updated_at;
    ALTER TABLE edge_devices ALTER COLUMN device_id SET NOT NULL;
    ALTER TABLE edge_devices ADD PRIMARY KEY (device_id);
  END IF;
END $$;
ALTER TABLE edge_devices ADD COLUMN IF NOT EXISTS site_name text;
ALTER TABLE edge_devices ADD COLUMN IF NOT EXISTS latitude  numeric;
ALTER TABLE edge_devices ADD COLUMN IF NOT EXISTS longitude numeric;

-- ============================================================================
-- PHASE 12: experiments FKs back, pointing to new text PKs
-- ============================================================================
DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname='experiments_user_id_fkey') THEN
    ALTER TABLE experiments
      ADD CONSTRAINT experiments_user_id_fkey
      FOREIGN KEY (user_id) REFERENCES users(username);
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname='experiments_edge_device_id_fkey') THEN
    ALTER TABLE experiments
      ADD CONSTRAINT experiments_edge_device_id_fkey
      FOREIGN KEY (edge_device_id) REFERENCES edge_devices(device_id);
  END IF;
END $$;

-- ============================================================================
-- PHASE 13: models cleanup
-- ============================================================================
ALTER TABLE models DROP CONSTRAINT IF EXISTS models_ckn_model_id_key;
ALTER TABLE models DROP CONSTRAINT IF EXISTS models_model_uid_key;
ALTER TABLE models DROP CONSTRAINT IF EXISTS models_name_key;
ALTER TABLE models DROP COLUMN IF EXISTS ckn_model_id;
ALTER TABLE models DROP COLUMN IF EXISTS model_uid;

-- ============================================================================
-- PHASE 14: Trigger functions + triggers
-- ============================================================================
CREATE OR REPLACE FUNCTION fn_ingest_camera_trap_event() RETURNS trigger AS $fn$
DECLARE
  v_model_id bigint;
  v_experiment_id bigint;
  v_raw_image_id bigint;
  v_scores jsonb;
BEGIN
  IF NOT EXISTS (SELECT 1 FROM users WHERE username = NEW.user_id) THEN
    RAISE EXCEPTION 'CKN ingest: user "%" not registered in patra (users.username)', NEW.user_id;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM edge_devices WHERE device_id = NEW.device_id) THEN
    RAISE EXCEPTION 'CKN ingest: edge_device "%" not registered in patra (edge_devices.device_id)', NEW.device_id;
  END IF;
  SELECT id INTO v_model_id FROM models WHERE id::text = NEW.model_id;
  IF v_model_id IS NULL THEN
    RAISE EXCEPTION 'CKN ingest: model "%" not registered in patra (models.id)', NEW.model_id;
  END IF;

  INSERT INTO experiments (
    experiment_uid, start_at, executed_at,
    total_images, total_predictions, total_ground_truth_objects,
    true_positives, false_positives, false_negatives,
    precision, recall, f1_score, mean_iou, map_50, map_50_95,
    user_id, edge_device_id, model_id
  ) VALUES (
    NEW.experiment_id,
    COALESCE(NEW.image_receiving_timestamp, NOW()),
    NEW.image_scoring_timestamp,
    NEW.total_images, NEW.total_predictions, NEW.total_ground_truth_objects,
    NEW.true_positives, NEW.false_positives, NEW.false_negatives,
    NEW.precision, NEW.recall, NEW.f1_score, NEW.mean_iou, NEW.map_50, NEW.map_50_95,
    NEW.user_id, NEW.device_id, v_model_id
  )
  ON CONFLICT ON CONSTRAINT experiments_experiment_uid_key DO UPDATE SET
    executed_at = GREATEST(experiments.executed_at, EXCLUDED.executed_at),
    total_images = EXCLUDED.total_images,
    total_predictions = EXCLUDED.total_predictions,
    total_ground_truth_objects = EXCLUDED.total_ground_truth_objects,
    true_positives = EXCLUDED.true_positives,
    false_positives = EXCLUDED.false_positives,
    false_negatives = EXCLUDED.false_negatives,
    precision = EXCLUDED.precision,
    recall = EXCLUDED.recall,
    f1_score = EXCLUDED.f1_score,
    mean_iou = EXCLUDED.mean_iou,
    map_50 = EXCLUDED.map_50,
    map_50_95 = EXCLUDED.map_50_95
  RETURNING id INTO v_experiment_id;

  INSERT INTO raw_images (image_uid, image_name, ground_truth)
  VALUES (
    NEW.uuid,
    COALESCE(NEW.image_name, NEW.uuid),
    CASE WHEN NEW.ground_truth IS NOT NULL
         THEN jsonb_build_object('label', NEW.ground_truth)
         ELSE NULL END
  )
  ON CONFLICT ON CONSTRAINT raw_images_image_uid_key DO UPDATE SET image_name = EXCLUDED.image_name
  RETURNING id INTO v_raw_image_id;

  v_scores := NULL;
  IF NEW.flattened_scores IS NOT NULL THEN
    BEGIN
      v_scores := NEW.flattened_scores::jsonb;
    EXCEPTION WHEN OTHERS THEN
      v_scores := jsonb_build_object('raw', NEW.flattened_scores);
    END;
  END IF;

  INSERT INTO experiment_images (
    experiment_id, raw_image_id, image_count,
    image_received_at, image_scored_at, image_store_deleted_at,
    image_decision, top_label, top_probability,
    ingested_at, scores
  ) VALUES (
    v_experiment_id, v_raw_image_id, COALESCE(NEW.image_count, 1),
    NEW.image_receiving_timestamp, NEW.image_scoring_timestamp, NEW.image_store_delete_time,
    NEW.image_decision, NEW.label, NEW.probability,
    COALESCE(NEW.ingested_at, NOW()), v_scores
  )
  ON CONFLICT (experiment_id, raw_image_id) DO UPDATE SET
    image_scored_at = EXCLUDED.image_scored_at,
    image_store_deleted_at = EXCLUDED.image_store_deleted_at,
    image_decision = EXCLUDED.image_decision,
    top_label = EXCLUDED.top_label,
    top_probability = EXCLUDED.top_probability,
    scores = EXCLUDED.scores;
  RETURN NEW;
END;
$fn$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION fn_ingest_camera_trap_power() RETURNS trigger AS $fn$
DECLARE
  v_experiment_id bigint;
BEGIN
  SELECT id INTO v_experiment_id
    FROM experiments
   WHERE experiment_uid = NEW.experiment_id;
  IF v_experiment_id IS NULL THEN
    RAISE NOTICE 'fn_ingest_camera_trap_power: experiment % not yet in normalized table; power deferred', NEW.experiment_id;
    RETURN NEW;
  END IF;
  UPDATE experiments
     SET total_cpu_power_w = NEW.total_cpu_power_consumption,
         total_gpu_power_w = NEW.total_gpu_power_consumption
   WHERE id = v_experiment_id;
  RETURN NEW;
END;
$fn$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_camera_trap_events_fanout ON events;
CREATE TRIGGER trg_camera_trap_events_fanout
  AFTER INSERT OR UPDATE ON events
  FOR EACH ROW EXECUTE FUNCTION fn_ingest_camera_trap_event();

DROP TRIGGER IF EXISTS trg_camera_trap_power_fanout ON power_summary;
CREATE TRIGGER trg_camera_trap_power_fanout
  AFTER INSERT OR UPDATE ON power_summary
  FOR EACH ROW EXECUTE FUNCTION fn_ingest_camera_trap_power();

-- ============================================================================
-- PHASE 15: Final verification
-- ============================================================================
DO $postflight$
DECLARE v_present text;
BEGIN
  SELECT string_agg(table_name || '.' || column_name, ', ') INTO v_present
  FROM information_schema.columns
  WHERE table_schema='public' AND (
    (table_name='users'        AND column_name IN ('id','ckn_user_id','created_at','updated_at')) OR
    (table_name='edge_devices' AND column_name IN ('id','created_at','updated_at'))               OR
    (table_name='models'       AND column_name IN ('ckn_model_id','model_uid'))                   OR
    (table_name='experiments'  AND column_name IN ('ckn_experiment_id','map_75','submitted_at',
                                                    'model_used_at','created_at','updated_at'))   OR
    (table_name='raw_images'   AND column_name IN ('ckn_uuid','created_at','updated_at'))         OR
    (table_name='model_cards'  AND column_name IN ('status','asset_version','previous_version_id',
                                                    'root_version_id','patra_mc_id'))             OR
    (table_name='datasheets'   AND column_name IN ('asset_version','previous_version_id',
                                                    'root_version_id','dataset_schema_id'))
  );
  IF v_present IS NOT NULL THEN
    RAISE EXCEPTION 'POSTFLIGHT FAIL: columns that should be dropped/renamed still present: %', v_present;
  END IF;
  RAISE NOTICE 'OK migration complete; all targeted columns/tables migrated';
END $postflight$;

COMMIT;
