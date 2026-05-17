-- Seed events table with realistic kafka fan-out data.
-- Pre-registers users + edge_devices, then inserts events that reference
-- existing models by id (CKN sends models.id as text).
--
-- Idempotent within itself: TRUNCATEs events/experiments/raw_images/
-- experiment_images first so re-running gives a clean state. Does NOT touch
-- model_cards or models.

BEGIN;

-- Wipe ingest-derived tables so re-running is clean.
TRUNCATE TABLE
  experiment_images, raw_images, experiments, events, power_summary
RESTART IDENTITY CASCADE;
DELETE FROM users WHERE username IN ('neelk', 'jstubbs', 'swithana', 'beckstei');
DELETE FROM edge_devices WHERE device_id IN (
  'jetson-nano-iu-01', 'rpi4-camtrap-02', 'drone-mavic-01', 'cpu-x86-server'
);

-- ---------------------------------------------------------------------------
-- Pre-registered users
-- ---------------------------------------------------------------------------
INSERT INTO users (username) VALUES
  ('neelk'),
  ('jstubbs'),
  ('swithana'),
  ('beckstei');

-- ---------------------------------------------------------------------------
-- Pre-registered edge_devices
-- ---------------------------------------------------------------------------
INSERT INTO edge_devices (device_id, device_type, site_name, latitude, longitude) VALUES
  ('jetson-nano-iu-01', 'Jetson Nano',  'IU Bloomington Lab',                39.1729, -86.5263),
  ('rpi4-camtrap-02',   'Raspberry Pi 4','Morgan-Monroe State Forest',       39.2920, -86.4150),
  ('drone-mavic-01',    'DJI Mavic 3',  'Purdue Agronomy Research Station',  40.4830, -86.9930),
  ('cpu-x86-server',    'CPU x86_64',   'TACC Frontera, Austin TX',          30.3892, -97.7268);

-- ---------------------------------------------------------------------------
-- Events. Each row triggers fanout into experiments/raw_images/experiment_images.
-- model_id is the text representation of the patra-assigned models.id.
-- We use models 1 (ResNet50), 2 (MegaDetector), 7 (Yolo-SoftToyWeb), 16 (GoogLeNet).
-- ---------------------------------------------------------------------------

-- Experiment A: ResNet50 classification @ Jetson Nano (5 images)
INSERT INTO events (
  uuid, image_count, image_name, ground_truth,
  image_receiving_timestamp, image_scoring_timestamp, image_decision,
  model_id, label, probability, flattened_scores,
  device_id, experiment_id, user_id,
  total_images, total_predictions, total_ground_truth_objects,
  true_positives, false_positives, false_negatives,
  precision, recall, f1_score, mean_iou, map_50, map_50_95
) VALUES
  ('expA-img-001', 1, 'IMG_20260510_080112.jpg', 'cat',
   '2026-05-10 08:01:12+00','2026-05-10 08:01:13+00','Save',
   '1', 'cat', 0.9410000,
   '[{"label":"cat","probability":0.941},{"label":"dog","probability":0.038}]',
   'jetson-nano-iu-01','exp-resnet50-cat-classify-1','neelk',
   5,5,4, 4,1,0, 0.80000,1.00000,0.88889, 0.66800,0.81100,0.69900),

  ('expA-img-002', 2, 'IMG_20260510_081245.jpg', 'dog',
   '2026-05-10 08:12:45+00','2026-05-10 08:12:46+00','Save',
   '1', 'dog', 0.8870000,
   '[{"label":"dog","probability":0.887},{"label":"cat","probability":0.083}]',
   'jetson-nano-iu-01','exp-resnet50-cat-classify-1','neelk',
   5,5,4, 4,1,0, 0.80000,1.00000,0.88889, 0.66800,0.81100,0.69900),

  ('expA-img-003', 3, 'IMG_20260510_083010.jpg', 'cat',
   '2026-05-10 08:30:10+00','2026-05-10 08:30:11+00','Save',
   '1', 'cat', 0.9210000,
   '[{"label":"cat","probability":0.921}]',
   'jetson-nano-iu-01','exp-resnet50-cat-classify-1','neelk',
   5,5,4, 4,1,0, 0.80000,1.00000,0.88889, 0.66800,0.81100,0.69900);

-- Experiment B: MegaDetector wildlife detection @ RPi4 camera trap (4 images)
INSERT INTO events (
  uuid, image_count, image_name, ground_truth,
  image_receiving_timestamp, image_scoring_timestamp, image_decision,
  model_id, label, probability, flattened_scores,
  device_id, experiment_id, user_id,
  total_images, total_predictions, total_ground_truth_objects,
  true_positives, false_positives, false_negatives,
  precision, recall, f1_score, mean_iou, map_50, map_50_95
) VALUES
  ('expB-img-001', 1, 'CAM02_20260512_063201.jpg', 'deer',
   '2026-05-12 06:32:01+00','2026-05-12 06:32:03+00','Save',
   '2', 'animal', 0.9870000,
   '[{"label":"animal","probability":0.987},{"label":"empty","probability":0.008}]',
   'rpi4-camtrap-02','exp-megadetector-wildlife-may','jstubbs',
   4,4,4, 3,1,1, 0.75000,0.75000,0.75000, 0.71200,0.85600,0.74500),

  ('expB-img-002', 2, 'CAM02_20260512_071145.jpg', 'empty',
   '2026-05-12 07:11:45+00','2026-05-12 07:11:47+00','Discard',
   '2', 'empty', 0.9210000,
   '[{"label":"empty","probability":0.921},{"label":"animal","probability":0.067}]',
   'rpi4-camtrap-02','exp-megadetector-wildlife-may','jstubbs',
   4,4,4, 3,1,1, 0.75000,0.75000,0.75000, 0.71200,0.85600,0.74500),

  ('expB-img-003', 3, 'CAM02_20260512_193002.jpg', 'raccoon',
   '2026-05-12 19:30:02+00','2026-05-12 19:30:04+00','Save',
   '2', 'animal', 0.8910000,
   '[{"label":"animal","probability":0.891}]',
   'rpi4-camtrap-02','exp-megadetector-wildlife-may','jstubbs',
   4,4,4, 3,1,1, 0.75000,0.75000,0.75000, 0.71200,0.85600,0.74500),

  ('expB-img-004', 4, 'CAM02_20260513_021533.jpg', 'coyote',
   '2026-05-13 02:15:33+00','2026-05-13 02:15:35+00','Save',
   '2', 'animal', 0.9450000,
   '[{"label":"animal","probability":0.945}]',
   'rpi4-camtrap-02','exp-megadetector-wildlife-may','jstubbs',
   4,4,4, 3,1,1, 0.75000,0.75000,0.75000, 0.71200,0.85600,0.74500);

-- Experiment C: YOLO toy detection @ datacenter (3 images)
INSERT INTO events (
  uuid, image_count, image_name, ground_truth,
  image_receiving_timestamp, image_scoring_timestamp, image_decision,
  model_id, label, probability, flattened_scores,
  device_id, experiment_id, user_id,
  total_images, total_predictions, total_ground_truth_objects,
  true_positives, false_positives, false_negatives,
  precision, recall, f1_score, mean_iou, map_50, map_50_95
) VALUES
  ('expC-img-001', 1, 'TOY_20260514_140012.jpg', 'plush_bear',
   '2026-05-14 14:00:12+00','2026-05-14 14:00:13+00','Save',
   '7', 'plush_bear', 0.9650000,
   '[{"label":"plush_bear","probability":0.965}]',
   'cpu-x86-server','exp-yolo-toyweb-validation','swithana',
   3,3,3, 3,0,0, 1.00000,1.00000,1.00000, 0.78900,0.92100,0.81200),

  ('expC-img-002', 2, 'TOY_20260514_141234.jpg', 'plush_dog',
   '2026-05-14 14:12:34+00','2026-05-14 14:12:35+00','Save',
   '7', 'plush_dog', 0.9510000,
   '[{"label":"plush_dog","probability":0.951}]',
   'cpu-x86-server','exp-yolo-toyweb-validation','swithana',
   3,3,3, 3,0,0, 1.00000,1.00000,1.00000, 0.78900,0.92100,0.81200),

  ('expC-img-003', 3, 'TOY_20260514_143510.jpg', 'plush_rabbit',
   '2026-05-14 14:35:10+00','2026-05-14 14:35:12+00','Save',
   '7', 'plush_rabbit', 0.9320000,
   '[{"label":"plush_rabbit","probability":0.932}]',
   'cpu-x86-server','exp-yolo-toyweb-validation','swithana',
   3,3,3, 3,0,0, 1.00000,1.00000,1.00000, 0.78900,0.92100,0.81200);

-- Experiment D: GoogLeNet drone crop survey (2 images)
INSERT INTO events (
  uuid, image_count, image_name, ground_truth,
  image_receiving_timestamp, image_scoring_timestamp, image_decision,
  model_id, label, probability, flattened_scores,
  device_id, experiment_id, user_id,
  total_images, total_predictions, total_ground_truth_objects,
  true_positives, false_positives, false_negatives,
  precision, recall, f1_score, mean_iou, map_50, map_50_95
) VALUES
  ('expD-img-001', 1, 'DRONE_20260515_102301.jpg', 'corn_healthy',
   '2026-05-15 10:23:01+00','2026-05-15 10:23:04+00','Save',
   '16', 'corn_healthy', 0.9720000,
   '[{"label":"corn_healthy","probability":0.972}]',
   'drone-mavic-01','exp-googlenet-cornfield-survey','beckstei',
   2,2,2, 2,0,0, 1.00000,1.00000,1.00000, 0.81000,0.88500,0.78400),

  ('expD-img-002', 2, 'DRONE_20260515_104015.jpg', 'corn_diseased',
   '2026-05-15 10:40:15+00','2026-05-15 10:40:18+00','Save',
   '16', 'corn_diseased', 0.8430000,
   '[{"label":"corn_diseased","probability":0.843},{"label":"corn_healthy","probability":0.121}]',
   'drone-mavic-01','exp-googlenet-cornfield-survey','beckstei',
   2,2,2, 2,0,0, 1.00000,1.00000,1.00000, 0.81000,0.88500,0.78400);

-- ---------------------------------------------------------------------------
-- Power summaries (one row per experiment_uid, fanned out by power trigger)
-- ---------------------------------------------------------------------------
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
  ('exp-resnet50-cat-classify-1',   1.2, 4.0, 0.6, 0.3, 0.8, 5.5,  2.6,  9.8),
  ('exp-megadetector-wildlife-may', 0.9, 0.0, 0.4, 0.0, 0.7, 0.0,  2.0,  0.0),
  ('exp-yolo-toyweb-validation',   12.5, 92.0, 4.1, 2.0, 8.7, 110.5, 25.3, 204.5),
  ('exp-googlenet-cornfield-survey', 6.8, 32.5, 2.0, 1.2, 5.5, 41.0, 14.3, 74.7);

COMMIT;
