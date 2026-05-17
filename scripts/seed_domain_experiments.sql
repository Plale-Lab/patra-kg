-- Seed synthetic rows for the /animal-ecology frontend domain page:
--   events, power_summary
--
-- Safe to run multiple times. It only replaces rows for known synthetic
-- experiment IDs used in this script.

BEGIN;

-- ---------------------------------------------------------------------------
-- Cleanup only this script's synthetic records
-- ---------------------------------------------------------------------------
DELETE FROM power_summary
WHERE experiment_id IN (
  'syn-animal-exp-001',
  'syn-animal-exp-002'
);

DELETE FROM events
WHERE experiment_id IN (
  'syn-animal-exp-001',
  'syn-animal-exp-002'
);

-- ---------------------------------------------------------------------------
-- Animal Ecology domain
-- ---------------------------------------------------------------------------
INSERT INTO events (
  uuid,
  image_count,
  image_name,
  ground_truth,
  image_receiving_timestamp,
  image_scoring_timestamp,
  image_store_delete_time,
  image_decision,
  model_id,
  label,
  probability,
  flattened_scores,
  device_id,
  experiment_id,
  user_id,
  total_images,
  total_predictions,
  total_ground_truth_objects,
  true_positives,
  false_positives,
  false_negatives,
  precision,
  recall,
  f1_score,
  mean_iou,
  map_50,
  map_50_95
) VALUES
  (
    'syn-cte-001',
    1,
    'animal_syn_001.jpg',
    'deer',
    NOW() - interval '6 hours',
    NOW() - interval '5 hours 59 minutes',
    NULL,
    'Save',
    'syn-megadetector-v1',
    'deer',
    0.9623000,
    '[{"label":"deer","probability":0.9623},{"label":"empty","probability":0.0214},{"label":"person","probability":0.0163}]',
    'jetson_orin_ae_01',
    'syn-animal-exp-001',
    'syn-user-ae',
    3,
    3,
    3,
    3,
    0,
    0,
    1.00000,
    1.00000,
    1.00000,
    0.78410,
    0.91340,
    0.80120
  ),
  (
    'syn-cte-002',
    2,
    'animal_syn_002.jpg',
    'fox',
    NOW() - interval '5 hours 30 minutes',
    NOW() - interval '5 hours 29 minutes',
    NULL,
    'Save',
    'syn-megadetector-v1',
    'fox',
    0.9081000,
    '[{"label":"fox","probability":0.9081},{"label":"deer","probability":0.0512},{"label":"empty","probability":0.0407}]',
    'jetson_orin_ae_01',
    'syn-animal-exp-001',
    'syn-user-ae',
    3,
    3,
    3,
    3,
    0,
    0,
    1.00000,
    1.00000,
    1.00000,
    0.78410,
    0.91340,
    0.80120
  ),
  (
    'syn-cte-003',
    3,
    'animal_syn_003.jpg',
    'empty',
    NOW() - interval '5 hours',
    NOW() - interval '4 hours 59 minutes',
    NULL,
    'Discard',
    'syn-megadetector-v1',
    'empty',
    0.8799000,
    '[{"label":"empty","probability":0.8799},{"label":"deer","probability":0.0701},{"label":"person","probability":0.0500}]',
    'jetson_orin_ae_01',
    'syn-animal-exp-001',
    'syn-user-ae',
    3,
    3,
    3,
    3,
    0,
    0,
    1.00000,
    1.00000,
    1.00000,
    0.78410,
    0.91340,
    0.80120
  ),
  (
    'syn-cte-004',
    1,
    'animal_syn_004.jpg',
    'boar',
    NOW() - interval '3 hours',
    NOW() - interval '2 hours 59 minutes',
    NULL,
    'Save',
    'syn-yolov9-wildlife',
    'boar',
    0.8910000,
    '[{"label":"boar","probability":0.8910},{"label":"deer","probability":0.0610},{"label":"empty","probability":0.0480}]',
    'rpi_cam_ae_02',
    'syn-animal-exp-002',
    'syn-user-ae',
    2,
    2,
    2,
    1,
    1,
    1,
    0.50000,
    0.50000,
    0.50000,
    0.62230,
    0.77440,
    0.65100
  ),
  (
    'syn-cte-005',
    2,
    'animal_syn_005.jpg',
    'empty',
    NOW() - interval '2 hours 40 minutes',
    NOW() - interval '2 hours 39 minutes',
    NULL,
    'Discard',
    'syn-yolov9-wildlife',
    'empty',
    0.8334000,
    '[{"label":"empty","probability":0.8334},{"label":"boar","probability":0.0956},{"label":"person","probability":0.0710}]',
    'rpi_cam_ae_02',
    'syn-animal-exp-002',
    'syn-user-ae',
    2,
    2,
    2,
    1,
    1,
    1,
    0.50000,
    0.50000,
    0.50000,
    0.62230,
    0.77440,
    0.65100
  );

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
) VALUES
  (
    'syn-animal-exp-001',
    11.2300,
    6.5400,
    1.8400,
    0.0000,
    17.9900,
    41.2000,
    31.0600,
    47.7400
  ),
  (
    'syn-animal-exp-002',
    8.7700,
    0.0000,
    1.3500,
    0.0000,
    13.4200,
    0.0000,
    23.5400,
    0.0000
  );


COMMIT;
