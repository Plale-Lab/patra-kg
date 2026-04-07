-- Production-like seed data for Patra Knowledge Graph (PostgreSQL)
--
-- Row counts:
--   model_cards: 10 (5 public, 5 private), models: 10, datasheets: 10
--   camera_trap_events: ~20, camera_trap_power: 3
--   digital_ag_events: 8, digital_ag_power: 2
--
-- Usage:
--   psql -d patra -f db/seed_synthetic_data.sql

BEGIN;

TRUNCATE TABLE
  digital_ag_power,
  digital_ag_events,
  camera_trap_power,
  camera_trap_events,
  experiment_images,
  experiments,
  raw_images,
  models,
  model_cards,
  dataset_schemas,
  edge_devices,
  users
RESTART IDENTITY CASCADE;

-- Ignore errors from tables that may not exist in this schema version
DO $$ BEGIN
  EXECUTE 'TRUNCATE TABLE deployments RESTART IDENTITY CASCADE';
EXCEPTION WHEN undefined_table THEN NULL;
END $$;

--------------------------------------------------------------------------------
-- 1. model_cards  (5 public + 5 private)
--------------------------------------------------------------------------------
INSERT INTO model_cards (
  name, version, is_private,
  short_description, full_description,
  keywords, author, citation,
  input_data, input_type, output_data,
  foundational_model, category, documentation,
  created_at, updated_at
) VALUES
  -- PUBLIC (5)
  ('HybridEnd2EndLearner', '5a', false,
   'HybridEnd2EndLearner 97% MNIST Dataset',
   'End-to-end hybrid quantum-classical learner achieving 97% accuracy on the MNIST handwritten digit dataset.',
   'hybrid,quantum,mnist,classification', 'nkarthikeyan',
   'Karthikeyan et al., 2024. HybridEnd2EndLearner for MNIST.',
   'MNIST 28x28 grayscale images', 'images', 'digit class probabilities',
   'HybridEnd2EndLearner', 'classification', '',
   NOW(), NOW()),

  ('MegaDetector for Wildlife Detection', '5a', false,
   'Wildlife detection using MegaDetector from Microsoft.',
   'MegaDetector v5a is a camera-trap image detector from Microsoft AI for Earth.',
   'wildlife,camera trap,megadetector,object detection', 'wqiu',
   'Beery et al., 2019. Efficient Pipeline for Camera Trap Image Review. arXiv:1907.06772',
   'Camera trap images', 'images', 'Bounding boxes (animal, person, vehicle)',
   'MegaDetector', 'detection', '',
   NOW(), NOW()),

  ('GoogLeNet for Image Classification', '1.0', false,
   'Image classification using GoogLeNet.',
   'GoogLeNet (Inception v1) pre-trained on ImageNet for general-purpose image classification.',
   'googlenet,inception,classification,imagenet', 'jstubbs',
   'Szegedy et al., 2015. Going Deeper with Convolutions. CVPR 2015.',
   'https://image-net.org/', 'images', 'ImageNet class probabilities',
   'GoogLeNet', 'classification', '',
   NOW(), NOW()),

  ('ResNet50 Image Classification Model', '1.0', false,
   'Pre-trained ResNet50 model from torchvision for image classification.',
   'Pre-trained ResNet50 from torchvision, fine-tuned for general image classification benchmarks.',
   'resnet50,classification,torchvision,imagenet', 'cgarcia',
   'He et al., 2016. Deep Residual Learning for Image Recognition. CVPR 2016.',
   'https://image-net.org/', 'images', 'ImageNet class probabilities',
   'ResNet50', 'classification', '',
   NOW(), NOW()),

  ('Ultralytics YOLO', '9e', false,
   'YOLOv9',
   'YOLOv9 marks a significant advancement in real-time object detection, introducing PGI and GELAN.',
   'yolo, ultralytics, object detection', 'skhuvis',
   'YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information (arXiv:2402.13616)',
   'https://cocodataset.org/', 'images', '',
   'YOLOv5', 'classification', '',
   NOW(), NOW()),

  -- PRIVATE (5)
  ('MegaDetector for Wildlife Detection', '5a (OSA finetuning)', true,
   'Wildlife detection using MegaDetector from Microsoft.',
   'MegaDetector v5a fine-tuned on the OSA (Open Science Alliance) camera trap dataset.',
   'wildlife,camera trap,megadetector,finetuned,osa', 'rcardone',
   'Beery et al., 2019. Efficient Pipeline for Camera Trap Image Review. arXiv:1907.06772',
   'Camera trap images', 'images', 'Bounding boxes (animal, person, vehicle)',
   'MegaDetector', 'detection', '',
   NOW(), NOW()),

  ('MegaDetector for Wildlife Detection', '6b-yolov9c', true,
   'Wildlife detection using MegaDetector from Microsoft.',
   'MegaDetector v6b built on YOLOv9c backbone for improved speed-accuracy trade-off.',
   'wildlife,camera trap,megadetector,yolov9', 'wqiu',
   'Beery et al., 2019. Efficient Pipeline for Camera Trap Image Review. arXiv:1907.06772',
   'Camera trap images', 'images', 'Bounding boxes (animal, person, vehicle)',
   'YOLOv9c', 'detection', '',
   NOW(), NOW()),

  ('Yolo Object Detecion - for detecting a soft toy',
   'yolo11l_ep1_bs32_lr0.005_8aa95a86.pt', true,
   'Detecting Soft toys in frame usining fine-tuned yolo model',
   'YOLOv11-large fine-tuned to detect soft toys in camera frames for inventory counting.',
   'yolo,object detection,soft toy,inventory', 'skhuvis',
   '', 'Camera frames', 'images', 'Bounding boxes with confidence',
   'YOLOv11', 'detection', '',
   NOW(), NOW()),

  ('Ultralytics YOLO26n', '6b-yolov9c', true,
   'YOLO26n.',
   'YOLO26 is the latest evolution in the YOLO series, engineered for edge and low-power devices.',
   'yolo, ultralytics, object detection', 'skhuvis',
   'https://huggingface.co/Ultralytics/YOLO26',
   'https://cocodataset.org/', 'images', '',
   'YOLO26', 'classification', '',
   NOW(), NOW()),

  ('Ultralytics YOLO26x', '26x', true,
   'YOLO26x.',
   'YOLO26x is the extra-large variant of YOLO26 for maximum accuracy on high-resolution inputs.',
   'yolo, ultralytics, object detection, yolo26', 'skhuvis',
   'https://huggingface.co/Ultralytics/YOLO26',
   'https://cocodataset.org/', 'images', '',
   'YOLO26', 'classification', '',
   NOW(), NOW());

--------------------------------------------------------------------------------
-- 2. models  (1:1 with model_cards — use generated IDs)
--------------------------------------------------------------------------------
INSERT INTO models (
  name, version, description, owner, location, license,
  framework, model_type, test_accuracy, model_card_id,
  created_at, updated_at
)
SELECT
  mc.name, mc.version,
  mc.full_description, mc.author, '', '',
  'PyTorch', 'convolutional neural network', 0.85,
  mc.id, NOW(), NOW()
FROM model_cards mc;

--------------------------------------------------------------------------------
-- 3. datasheets  (5 public + 5 private)
--------------------------------------------------------------------------------
INSERT INTO datasheets (
  publication_year, resource_type, size, format, version,
  is_private, created_at, updated_at
) VALUES
  (2021, 'images', '3 TB', 'jpeg', '1.0', false, NOW(), NOW()),
  (2012, 'images', '150 GB', 'jpeg', '1.0', false, NOW(), NOW()),
  (1998, 'images', '50 MB', 'idx', '1.0', false, NOW(), NOW()),
  (2017, 'images', '25 GB', 'jpeg', '2017', false, NOW(), NOW()),
  (2015, 'images', '12 GB', 'jpeg', '1.0', false, NOW(), NOW()),
  (2024, 'images', '18 GB', 'jpeg', '1.0', true, NOW(), NOW()),
  (2024, 'images', '32 GB', 'jpeg', '1.0', true, NOW(), NOW()),
  (2025, 'images', '2 GB', 'jpeg', '1.0', true, NOW(), NOW()),
  (2025, 'images', '8 GB', 'jpeg', '1.0', true, NOW(), NOW()),
  (2025, 'images', '45 GB', 'png', '1.0', true, NOW(), NOW());

--------------------------------------------------------------------------------
-- 4. camera_trap_events  (flat table — mirrors Kafka oracle-events topic)
--------------------------------------------------------------------------------
INSERT INTO camera_trap_events (
  uuid, image_count, image_name, ground_truth,
  image_receiving_timestamp, image_scoring_timestamp, image_store_delete_time,
  image_decision, model_id, label, probability, flattened_scores,
  device_id, experiment_id, user_id,
  total_images, total_predictions, total_ground_truth_objects,
  true_positives, false_positives, false_negatives,
  precision, recall, f1_score, mean_iou, map_50, map_50_95
) VALUES
  -- Experiment: googlenet-iu-animal-classification (user: jstubbs)
  ('evt-001', 1, 'IMG_20260320_081023.jpg', 'deer',
   '2026-03-20 08:10:23+00', '2026-03-20 08:10:24+00', NULL,
   'Save', 'googlenet-iu-animal', 'deer', 0.9430000,
   '[{"label":"deer","probability":0.943},{"label":"empty","probability":0.036},{"label":"person","probability":0.021}]',
   'jetson_nano_01', 'googlenet-iu-animal-classification', 'jstubbs',
   5, 5, 4, 4, 1, 0, 0.80000, 1.00000, 0.88889, 0.67800, 0.82300, 0.71200),

  ('evt-002', 2, 'IMG_20260320_083512.jpg', 'empty',
   '2026-03-20 08:35:12+00', '2026-03-20 08:35:13+00', NULL,
   'Discard', 'googlenet-iu-animal', 'empty', 0.8710000,
   '[{"label":"empty","probability":0.871},{"label":"deer","probability":0.089},{"label":"person","probability":0.040}]',
   'jetson_nano_01', 'googlenet-iu-animal-classification', 'jstubbs',
   5, 5, 4, 4, 1, 0, 0.80000, 1.00000, 0.88889, 0.67800, 0.82300, 0.71200),

  ('evt-003', 3, 'IMG_20260320_091204.jpg', 'coyote',
   '2026-03-20 09:12:04+00', '2026-03-20 09:12:05+00', NULL,
   'Save', 'googlenet-iu-animal', 'coyote', 0.9670000,
   '[{"label":"coyote","probability":0.967},{"label":"empty","probability":0.021},{"label":"person","probability":0.012}]',
   'jetson_nano_01', 'googlenet-iu-animal-classification', 'jstubbs',
   5, 5, 4, 4, 1, 0, 0.80000, 1.00000, 0.88889, 0.67800, 0.82300, 0.71200),

  ('evt-004', 4, 'IMG_20260320_102345.jpg', 'person',
   '2026-03-20 10:23:45+00', '2026-03-20 10:23:46+00', NULL,
   'Save', 'googlenet-iu-animal', 'person', 0.8120000,
   '[{"label":"person","probability":0.812},{"label":"deer","probability":0.102},{"label":"empty","probability":0.086}]',
   'jetson_nano_01', 'googlenet-iu-animal-classification', 'jstubbs',
   5, 5, 4, 4, 1, 0, 0.80000, 1.00000, 0.88889, 0.67800, 0.82300, 0.71200),

  ('evt-005', 5, 'IMG_20260320_114502.jpg', 'empty',
   '2026-03-20 11:45:02+00', '2026-03-20 11:45:03+00', NULL,
   'Discard', 'googlenet-iu-animal', 'empty', 0.9340000,
   '[{"label":"empty","probability":0.934},{"label":"person","probability":0.034},{"label":"deer","probability":0.032}]',
   'jetson_nano_01', 'googlenet-iu-animal-classification', 'jstubbs',
   5, 5, 4, 4, 1, 0, 0.80000, 1.00000, 0.88889, 0.67800, 0.82300, 0.71200),

  -- Experiment: megadetector-rpi-wildlife (user: jstubbs)
  ('evt-006', 1, 'CAM02_20260322_143200.jpg', 'raccoon',
   '2026-03-22 14:32:00+00', '2026-03-22 14:32:01+00', NULL,
   'Save', 'megadetector-v5a', 'animal', 0.9890000,
   '[{"label":"animal","probability":0.989},{"label":"empty","probability":0.006},{"label":"person","probability":0.005}]',
   'rpi4_cam_01', 'megadetector-rpi-wildlife', 'jstubbs',
   3, 3, 3, 2, 1, 1, 0.66667, 0.66667, 0.66667, 0.71200, 0.85600, 0.74500),

  ('evt-007', 2, 'CAM02_20260322_151045.jpg', 'deer',
   '2026-03-22 15:10:45+00', '2026-03-22 15:10:46+00', NULL,
   'Save', 'megadetector-v5a', 'animal', 0.8760000,
   '[{"label":"animal","probability":0.876},{"label":"person","probability":0.067},{"label":"empty","probability":0.057}]',
   'rpi4_cam_01', 'megadetector-rpi-wildlife', 'jstubbs',
   3, 3, 3, 2, 1, 1, 0.66667, 0.66667, 0.66667, 0.71200, 0.85600, 0.74500),

  ('evt-008', 3, 'CAM02_20260322_161200.jpg', 'empty',
   '2026-03-22 16:12:00+00', '2026-03-22 16:12:01+00', NULL,
   'Discard', 'megadetector-v5a', 'empty', 0.9120000,
   '[{"label":"empty","probability":0.912},{"label":"animal","probability":0.055},{"label":"person","probability":0.033}]',
   'rpi4_cam_01', 'megadetector-rpi-wildlife', 'jstubbs',
   3, 3, 3, 2, 1, 1, 0.66667, 0.66667, 0.66667, 0.71200, 0.85600, 0.74500),

  -- Experiment: yolov9-wildlife-survey (user: wqiu)
  ('evt-009', 1, 'WILD_20260318_060512.jpg', 'bear',
   '2026-03-18 06:05:12+00', '2026-03-18 06:05:13+00', NULL,
   'Save', 'yolov9-wildlife', 'animal', 0.9210000,
   '[{"label":"animal","probability":0.921},{"label":"person","probability":0.045},{"label":"empty","probability":0.034}]',
   'coral_tpu_01', 'yolov9-wildlife-survey', 'wqiu',
   4, 4, 3, 3, 1, 0, 0.75000, 1.00000, 0.85714, 0.62300, 0.76500, 0.65100),

  ('evt-010', 2, 'WILD_20260318_072301.jpg', 'empty',
   '2026-03-18 07:23:01+00', '2026-03-18 07:23:02+00', NULL,
   'Discard', 'yolov9-wildlife', 'empty', 0.7560000,
   '[{"label":"empty","probability":0.756},{"label":"animal","probability":0.178},{"label":"person","probability":0.066}]',
   'coral_tpu_01', 'yolov9-wildlife-survey', 'wqiu',
   4, 4, 3, 3, 1, 0, 0.75000, 1.00000, 0.85714, 0.62300, 0.76500, 0.65100),

  ('evt-011', 3, 'WILD_20260318_093045.jpg', 'deer',
   '2026-03-18 09:30:45+00', '2026-03-18 09:30:46+00', NULL,
   'Save', 'yolov9-wildlife', 'animal', 0.8870000,
   '[{"label":"animal","probability":0.887},{"label":"empty","probability":0.078},{"label":"person","probability":0.035}]',
   'coral_tpu_01', 'yolov9-wildlife-survey', 'wqiu',
   4, 4, 3, 3, 1, 0, 0.75000, 1.00000, 0.85714, 0.62300, 0.76500, 0.65100),

  ('evt-012', 4, 'WILD_20260318_110230.jpg', 'turkey',
   '2026-03-18 11:02:30+00', '2026-03-18 11:02:31+00', NULL,
   'Save', 'yolov9-wildlife', 'animal', 0.9450000,
   '[{"label":"animal","probability":0.945},{"label":"empty","probability":0.032},{"label":"person","probability":0.023}]',
   'coral_tpu_01', 'yolov9-wildlife-survey', 'wqiu',
   4, 4, 3, 3, 1, 0, 0.75000, 1.00000, 0.85714, 0.62300, 0.76500, 0.65100),

  -- Experiment: efficientdet-coral-campus (user: nkarthikeyan)
  ('evt-013', 1, 'CORAL_20260325_101530.jpg', 'squirrel',
   '2026-03-25 10:15:30+00', '2026-03-25 10:15:31+00', NULL,
   'Save', 'efficientdet-lite', 'animal', 0.9780000,
   '[{"label":"animal","probability":0.978},{"label":"empty","probability":0.011},{"label":"person","probability":0.011}]',
   'coral_tpu_01', 'efficientdet-coral-campus', 'nkarthikeyan',
   2, 2, 2, 2, 0, 0, 1.00000, 1.00000, 1.00000, 0.75600, 0.89100, 0.78900),

  ('evt-014', 2, 'CORAL_20260325_112045.jpg', 'rabbit',
   '2026-03-25 11:20:45+00', '2026-03-25 11:20:46+00', NULL,
   'Save', 'efficientdet-lite', 'animal', 0.9340000,
   '[{"label":"animal","probability":0.934},{"label":"person","probability":0.041},{"label":"empty","probability":0.025}]',
   'coral_tpu_01', 'efficientdet-coral-campus', 'nkarthikeyan',
   2, 2, 2, 2, 0, 0, 1.00000, 1.00000, 1.00000, 0.75600, 0.89100, 0.78900);

--------------------------------------------------------------------------------
-- 5. camera_trap_power  (flat table — mirrors Kafka power-summary topic)
--------------------------------------------------------------------------------
INSERT INTO camera_trap_power (
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
  ('googlenet-iu-animal-classification', 12.3400, 8.5600, 2.1000, 0.0000, 18.7800, 45.2300, 33.2200, 53.7900),
  ('megadetector-rpi-wildlife', 8.9100, 0.0000, 1.5600, 0.0000, 14.2300, 0.0000, 24.7000, 0.0000),
  ('yolov9-wildlife-survey', 15.6700, 22.3400, 3.4500, 1.2300, 25.8900, 67.4500, 45.0100, 91.0200);

--------------------------------------------------------------------------------
-- 6. digital_ag_events  (flat table — digital agriculture domain)
--------------------------------------------------------------------------------
INSERT INTO digital_ag_events (
  uuid, image_count, image_name, ground_truth,
  image_receiving_timestamp, image_scoring_timestamp, image_store_delete_time,
  image_decision, model_id, label, probability, flattened_scores,
  device_id, experiment_id, user_id,
  total_images, total_predictions, total_ground_truth_objects,
  true_positives, false_positives, false_negatives,
  precision, recall, f1_score, mean_iou, map_50, map_50_95
) VALUES
  -- Experiment: drone-corn-field-survey (user: cgarcia)
  ('dag-001', 1, 'DRONE_20260401_090100.jpg', 'healthy',
   '2026-04-01 09:01:00+00', '2026-04-01 09:01:02+00', NULL,
   'Save', 'yolov9-crop', 'healthy', 0.9510000,
   '[{"label":"healthy","probability":0.951},{"label":"stressed","probability":0.032},{"label":"diseased","probability":0.017}]',
   'dji_mavic_01', 'drone-corn-field-survey', 'cgarcia',
   4, 4, 4, 3, 1, 1, 0.75000, 0.75000, 0.75000, 0.69200, 0.81400, 0.70100),

  ('dag-002', 2, 'DRONE_20260401_091530.jpg', 'stressed',
   '2026-04-01 09:15:30+00', '2026-04-01 09:15:32+00', NULL,
   'Save', 'yolov9-crop', 'stressed', 0.8830000,
   '[{"label":"stressed","probability":0.883},{"label":"healthy","probability":0.078},{"label":"diseased","probability":0.039}]',
   'dji_mavic_01', 'drone-corn-field-survey', 'cgarcia',
   4, 4, 4, 3, 1, 1, 0.75000, 0.75000, 0.75000, 0.69200, 0.81400, 0.70100),

  ('dag-003', 3, 'DRONE_20260401_093200.jpg', 'diseased',
   '2026-04-01 09:32:00+00', '2026-04-01 09:32:02+00', NULL,
   'Save', 'yolov9-crop', 'diseased', 0.9120000,
   '[{"label":"diseased","probability":0.912},{"label":"stressed","probability":0.056},{"label":"healthy","probability":0.032}]',
   'dji_mavic_01', 'drone-corn-field-survey', 'cgarcia',
   4, 4, 4, 3, 1, 1, 0.75000, 0.75000, 0.75000, 0.69200, 0.81400, 0.70100),

  ('dag-004', 4, 'DRONE_20260401_100500.jpg', 'healthy',
   '2026-04-01 10:05:00+00', '2026-04-01 10:05:02+00', NULL,
   'Discard', 'yolov9-crop', 'healthy', 0.7450000,
   '[{"label":"healthy","probability":0.745},{"label":"stressed","probability":0.167},{"label":"diseased","probability":0.088}]',
   'dji_mavic_01', 'drone-corn-field-survey', 'cgarcia',
   4, 4, 4, 3, 1, 1, 0.75000, 0.75000, 0.75000, 0.69200, 0.81400, 0.70100),

  -- Experiment: satellite-soybean-yield (user: rcardone)
  ('dag-005', 1, 'SAT_20260328_120000.tif', 'high_yield',
   '2026-03-28 12:00:00+00', '2026-03-28 12:00:05+00', NULL,
   'Save', 'resnet50-yield', 'high_yield', 0.9340000,
   '[{"label":"high_yield","probability":0.934},{"label":"medium_yield","probability":0.045},{"label":"low_yield","probability":0.021}]',
   'sentinel2_tile', 'satellite-soybean-yield', 'rcardone',
   4, 4, 3, 3, 1, 0, 0.75000, 1.00000, 0.85714, 0.71500, 0.83200, 0.72400),

  ('dag-006', 2, 'SAT_20260328_123000.tif', 'medium_yield',
   '2026-03-28 12:30:00+00', '2026-03-28 12:30:05+00', NULL,
   'Save', 'resnet50-yield', 'medium_yield', 0.8670000,
   '[{"label":"medium_yield","probability":0.867},{"label":"high_yield","probability":0.089},{"label":"low_yield","probability":0.044}]',
   'sentinel2_tile', 'satellite-soybean-yield', 'rcardone',
   4, 4, 3, 3, 1, 0, 0.75000, 1.00000, 0.85714, 0.71500, 0.83200, 0.72400),

  ('dag-007', 3, 'SAT_20260328_130000.tif', 'low_yield',
   '2026-03-28 13:00:00+00', '2026-03-28 13:00:05+00', NULL,
   'Save', 'resnet50-yield', 'low_yield', 0.9010000,
   '[{"label":"low_yield","probability":0.901},{"label":"medium_yield","probability":0.067},{"label":"high_yield","probability":0.032}]',
   'sentinel2_tile', 'satellite-soybean-yield', 'rcardone',
   4, 4, 3, 3, 1, 0, 0.75000, 1.00000, 0.85714, 0.71500, 0.83200, 0.72400),

  ('dag-008', 4, 'SAT_20260328_133000.tif', 'high_yield',
   '2026-03-28 13:30:00+00', '2026-03-28 13:30:05+00', NULL,
   'Save', 'resnet50-yield', 'high_yield', 0.8920000,
   '[{"label":"high_yield","probability":0.892},{"label":"medium_yield","probability":0.072},{"label":"low_yield","probability":0.036}]',
   'sentinel2_tile', 'satellite-soybean-yield', 'rcardone',
   4, 4, 3, 3, 1, 0, 0.75000, 1.00000, 0.85714, 0.71500, 0.83200, 0.72400);

--------------------------------------------------------------------------------
-- 7. digital_ag_power  (flat table — digital agriculture power data)
--------------------------------------------------------------------------------
INSERT INTO digital_ag_power (
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
  ('drone-corn-field-survey', 10.2300, 15.6700, 1.8900, 0.5600, 20.4500, 52.3400, 32.5700, 68.5700),
  ('satellite-soybean-yield', 5.6700, 0.0000, 1.2300, 0.0000, 12.3400, 0.0000, 19.2400, 0.0000);

COMMIT;
