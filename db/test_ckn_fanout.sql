-- ============================================================================
-- Verification for the CKN → Patra normalized-table fan-out triggers.
-- ============================================================================
-- Usage (once patra-backend has applied bootstrap_schema.sql):
--   psql -h localhost -U patra -d patra -f db/test_ckn_fanout.sql
--
-- Inserts a synthetic events row, then a power_summary row,
-- and asserts that the triggers fan each out into the normalized schema with
-- auto-created FK references. Then re-inserts an updated event for the same
-- experiment to confirm upsert (running aggregates updated, no duplicate rows).
--
-- Cleans up after itself so it's safe to run repeatedly.
-- ============================================================================

\set ON_ERROR_STOP on
\set test_exp_id 'test-exp-ckn-fanout-001'
\set test_uuid   'test-uuid-ckn-fanout-001'
\set test_user   'test-ckn-user'
\set test_device 'test-ckn-device'
\set test_model  'test-ckn-model'

BEGIN;

-- Clear any prior test run (in reverse FK order)
DELETE FROM experiment_images WHERE experiment_id IN (
  SELECT id FROM experiments WHERE ckn_experiment_id = :'test_exp_id');
DELETE FROM experiments WHERE ckn_experiment_id = :'test_exp_id';
DELETE FROM raw_images WHERE ckn_uuid = :'test_uuid';
DELETE FROM events WHERE experiment_id = :'test_exp_id';
DELETE FROM power_summary WHERE experiment_id = :'test_exp_id';
DELETE FROM models WHERE ckn_model_id = :'test_model';
DELETE FROM model_cards WHERE name = :'test_model' AND ckn_auto_created = true;
DELETE FROM edge_devices WHERE device_id = :'test_device';
DELETE FROM users WHERE username = :'test_user';

-- --------------------------------------------------------------------------
-- 1. First event: insert into events → trigger fans out
-- --------------------------------------------------------------------------
INSERT INTO events (
  uuid, image_count, image_name, ground_truth,
  image_receiving_timestamp, image_scoring_timestamp, image_store_delete_time,
  image_decision, model_id, label, probability, flattened_scores,
  device_id, experiment_id, user_id,
  total_images, total_predictions, total_ground_truth_objects,
  true_positives, false_positives, false_negatives,
  precision, recall, f1_score
) VALUES (
  :'test_uuid', 1, 'sample.jpg', 'deer',
  '2026-05-03T10:00:00Z', '2026-05-03T10:00:01Z', '2026-05-03T10:00:02Z',
  'Saved', :'test_model', 'deer', 0.92,
  '[{"label":"deer","probability":0.92}]',
  :'test_device', :'test_exp_id', :'test_user',
  10, 9, 10,
  9, 1, 1,
  0.90000, 0.90000, 0.90000
);

-- Assertions
DO $assert$
DECLARE
  v_count integer;
  v_val text;
BEGIN
  SELECT COUNT(*) INTO v_count FROM users WHERE username = 'test-ckn-user';
  IF v_count <> 1 THEN RAISE EXCEPTION 'users: expected 1, got %', v_count; END IF;

  SELECT COUNT(*) INTO v_count FROM edge_devices WHERE device_id = 'test-ckn-device';
  IF v_count <> 1 THEN RAISE EXCEPTION 'edge_devices: expected 1, got %', v_count; END IF;

  SELECT COUNT(*) INTO v_count FROM models WHERE ckn_model_id = 'test-ckn-model';
  IF v_count <> 1 THEN RAISE EXCEPTION 'models: expected 1, got %', v_count; END IF;

  SELECT COUNT(*) INTO v_count FROM model_cards WHERE name = 'test-ckn-model' AND ckn_auto_created = true;
  IF v_count <> 1 THEN RAISE EXCEPTION 'model_cards stub: expected 1, got %', v_count; END IF;

  SELECT COUNT(*) INTO v_count FROM experiments WHERE ckn_experiment_id = 'test-exp-ckn-fanout-001';
  IF v_count <> 1 THEN RAISE EXCEPTION 'experiments: expected 1, got %', v_count; END IF;

  SELECT COUNT(*) INTO v_count FROM raw_images WHERE ckn_uuid = 'test-uuid-ckn-fanout-001';
  IF v_count <> 1 THEN RAISE EXCEPTION 'raw_images: expected 1, got %', v_count; END IF;

  SELECT COUNT(*) INTO v_count FROM experiment_images ei
    JOIN experiments e ON ei.experiment_id = e.id
    WHERE e.ckn_experiment_id = 'test-exp-ckn-fanout-001';
  IF v_count <> 1 THEN RAISE EXCEPTION 'experiment_images: expected 1, got %', v_count; END IF;

  RAISE NOTICE 'Step 1 OK: auto-created users/edge_devices/models/model_cards + experiments/experiment_images/raw_images';
END
$assert$;

-- --------------------------------------------------------------------------
-- 2. Re-delivery: upsert the same event with updated running aggregates.
--    Tests idempotency — should UPDATE, not INSERT another experiment row.
-- --------------------------------------------------------------------------
INSERT INTO events (
  uuid, image_count, image_name, ground_truth,
  image_receiving_timestamp, image_scoring_timestamp, image_store_delete_time,
  image_decision, model_id, label, probability, flattened_scores,
  device_id, experiment_id, user_id,
  total_images, total_predictions, total_ground_truth_objects,
  true_positives, false_positives, false_negatives,
  precision, recall, f1_score
) VALUES (
  :'test_uuid', 1, 'sample.jpg', 'deer',
  '2026-05-03T10:00:00Z', '2026-05-03T10:00:05Z', '2026-05-03T10:00:06Z',
  'Saved', :'test_model', 'deer', 0.92,
  '[{"label":"deer","probability":0.92}]',
  :'test_device', :'test_exp_id', :'test_user',
  20, 18, 20,      -- updated totals
  18, 2, 2,
  0.90000, 0.90000, 0.90000
)
ON CONFLICT (uuid) DO UPDATE SET
  total_images = EXCLUDED.total_images,
  total_predictions = EXCLUDED.total_predictions,
  true_positives = EXCLUDED.true_positives,
  false_positives = EXCLUDED.false_positives,
  false_negatives = EXCLUDED.false_negatives;

DO $assert$
DECLARE
  v_count integer;
  v_total integer;
BEGIN
  SELECT COUNT(*) INTO v_count FROM experiments WHERE ckn_experiment_id = 'test-exp-ckn-fanout-001';
  IF v_count <> 1 THEN RAISE EXCEPTION 'experiments after re-delivery: expected 1, got %', v_count; END IF;

  SELECT total_images INTO v_total FROM experiments WHERE ckn_experiment_id = 'test-exp-ckn-fanout-001';
  IF v_total <> 20 THEN RAISE EXCEPTION 'experiments.total_images: expected 20, got %', v_total; END IF;

  SELECT COUNT(*) INTO v_count FROM users WHERE username = 'test-ckn-user';
  IF v_count <> 1 THEN RAISE EXCEPTION 'users after re-delivery: expected 1, got %', v_count; END IF;

  RAISE NOTICE 'Step 2 OK: upsert updated running aggregates without duplicating rows';
END
$assert$;

-- --------------------------------------------------------------------------
-- 3. Power summary: insert into power_summary → trigger updates experiment
-- --------------------------------------------------------------------------
INSERT INTO power_summary (
  experiment_id,
  image_generating_plugin_cpu_power_consumption,
  image_generating_plugin_gpu_power_consumption,
  power_monitor_plugin_cpu_power_consumption,
  power_monitor_plugin_gpu_power_consumption,
  image_scoring_plugin_cpu_power_consumption,
  image_scoring_plugin_gpu_power_consumption,
  total_cpu_power_consumption,
  total_gpu_power_consumption
) VALUES (
  :'test_exp_id',
  2.5, 0.08, 2.6, 0.07, 2.7, 0.09,
  7.8, 0.24
);

DO $assert$
DECLARE
  v_cpu numeric;
  v_gpu numeric;
BEGIN
  SELECT total_cpu_power_w, total_gpu_power_w INTO v_cpu, v_gpu
    FROM experiments WHERE ckn_experiment_id = 'test-exp-ckn-fanout-001';

  IF v_cpu <> 7.8000 THEN RAISE EXCEPTION 'experiments.total_cpu_power_w: expected 7.8, got %', v_cpu; END IF;
  IF v_gpu <> 0.2400 THEN RAISE EXCEPTION 'experiments.total_gpu_power_w: expected 0.24, got %', v_gpu; END IF;

  RAISE NOTICE 'Step 3 OK: power-summary trigger propagated into experiments.total_*_power_w';
END
$assert$;

-- --------------------------------------------------------------------------
-- 4. Cleanup so the test is re-runnable
-- --------------------------------------------------------------------------
DELETE FROM experiment_images WHERE experiment_id IN (
  SELECT id FROM experiments WHERE ckn_experiment_id = :'test_exp_id');
DELETE FROM experiments WHERE ckn_experiment_id = :'test_exp_id';
DELETE FROM raw_images WHERE ckn_uuid = :'test_uuid';
DELETE FROM events WHERE experiment_id = :'test_exp_id';
DELETE FROM power_summary WHERE experiment_id = :'test_exp_id';
DELETE FROM models WHERE ckn_model_id = :'test_model';
DELETE FROM model_cards WHERE name = :'test_model' AND ckn_auto_created = true;
DELETE FROM edge_devices WHERE device_id = :'test_device';
DELETE FROM users WHERE username = :'test_user';

COMMIT;

\echo ''
\echo '==================================================='
\echo 'CKN fan-out trigger tests: ALL ASSERTIONS PASSED'
\echo '==================================================='
