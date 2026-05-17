# Patra Model Card Schema

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `name` | string | yes | — | Model card name |
| `version` | string | yes | — | Model card version |
| `short_description` | string | yes | — | Brief description |
| `full_description` | string | yes | — | Comprehensive description |
| `author` | string | yes | — | Author or creator |
| `keywords` | string | yes | — | Search keywords |
| `input_data` | url / DOI / null | yes | — | Link to input dataset |
| `input_type` | string | yes | — | Type of input (e.g. `Image`, `Text`) |
| `output_data` | url / DOI / null | yes | — | Link to output dataset |
| `ai_model.name` | string | yes | — | Model name |
| `ai_model.version` | string | yes | — | Model version |
| `ai_model.description` | string | yes | — | Model description |
| `ai_model.owner` | string | yes | — | Model owner / org |
| `ai_model.location` | string | yes | — | Downloadable URL |
| `ai_model.license` | string | yes | — | License identifier (e.g. `Apache-2.0`) |
| `ai_model.framework` | enum | yes | — | `sklearn` · `tensorflow` · `pytorch` · `other` |
| `ai_model.test_accuracy` | number | yes | — | Test-set accuracy |
| `ai_model.model_type` | enum | yes | — | `cnn` · `decision_tree` · `dnn` · `rnn` · `svm` · `kmeans` · `llm` · `random_forest` · `lstm` · `gnn` · `other` |
| `ai_model.model_structure` | object | no | — | Free-form architecture details |
| `ai_model.model_metrics` | `{key, value}[]` | no | — | Additional named metrics |
| `ai_model.inference_labels` | string | no | — | Inference label(s) |
| `id` | string | no | — | Catalog-assigned ID |
| `category` | enum | no | — | `classification` · `regression` · `clustering` · `anomaly detection` · `dimensionality reduction` · `reinforcement learning` · `natural language processing` · `computer vision` · `recommendation systems` · `time series forecasting` · `graph learning` · `graph neural networks` · `generative modeling` · `transfer learning` · `self-supervised learning` · `semi-supervised learning` · `unsupervised learning` · `causal inference` · `multi-task learning` · `metric learning` · `density estimation` · `multi-label classification` · `ranking` · `structured prediction` · `neural architecture search` · `sequence modeling` · `embedding learning` · `other` |
| `citation` | string | no | — | Citation information |
| `documentation` | string | no | — | Documentation URL |
| `foundational_model` | string | no | — | ID of foundational model used |
| `is_private` | boolean | no | `false` | Hidden from public catalog search when `true` |
| `is_gated` | boolean | no | `false` | Model download requires approval; card stays discoverable |
| `model_requirements` | string[] / null | no | — | Dependency strings (e.g. `torch==2.1.0`) |
| `bias_analysis.demographic_parity_diff` | number | no | — | Demographic parity difference |
| `bias_analysis.equal_odds_difference` | number | no | — | Equalized odds difference |
| `xai_analysis.bias_metrics` | `{key, value}[]` | no | — | Explainability / bias scanner metrics |
| `runtime.experiments[].experiment_id` | string | required in item | — | CKN experiment identifier |
| `runtime.experiments[].device_id` | string | required in item | — | CKN edge device identifier |
| `runtime.experiments[].user_id` | string | required in item | — | CKN user identifier |
| `runtime.experiments[].model_id` | string | required in item | — | CKN model identifier |
| `runtime.experiments[].image_receiving_timestamp` | date-time / null | no | — | First image received |
| `runtime.experiments[].image_scoring_timestamp` | date-time / null | no | — | Most recent scoring event |
| `runtime.experiments[].total_images` | integer / null | no | — | Total images processed |
| `runtime.experiments[].total_predictions` | integer / null | no | — | Total predictions emitted |
| `runtime.experiments[].total_ground_truth_objects` | integer / null | no | — | Total ground-truth objects |
| `runtime.experiments[].true_positives` | integer / null | no | — | Correct positive predictions |
| `runtime.experiments[].false_positives` | integer / null | no | — | Incorrect positive predictions |
| `runtime.experiments[].false_negatives` | integer / null | no | — | Missed positives |
| `runtime.experiments[].precision` | number 0–1 / null | no | — | TP / (TP + FP) |
| `runtime.experiments[].recall` | number 0–1 / null | no | — | TP / (TP + FN) |
| `runtime.experiments[].f1_score` | number 0–1 / null | no | — | Harmonic mean of precision and recall |
| `runtime.experiments[].mean_iou` | number 0–1 / null | no | — | Mean IoU (detection models; null otherwise) |
| `runtime.experiments[].map_50` | number 0–1 / null | no | — | mAP @ IoU 0.50 |
| `runtime.experiments[].map_50_95` | number 0–1 / null | no | — | mAP @ IoU 0.50–0.95 |
| `runtime.experiments[].power_summary.image_generating_plugin_cpu_power_consumption` | number / null | no | — | Image generator CPU watts |
| `runtime.experiments[].power_summary.image_generating_plugin_gpu_power_consumption` | number / null | no | — | Image generator GPU watts |
| `runtime.experiments[].power_summary.power_monitor_plugin_cpu_power_consumption` | number / null | no | — | Power monitor CPU watts |
| `runtime.experiments[].power_summary.power_monitor_plugin_gpu_power_consumption` | number / null | no | — | Power monitor GPU watts |
| `runtime.experiments[].power_summary.image_scoring_plugin_cpu_power_consumption` | number / null | no | — | Scoring plugin CPU watts |
| `runtime.experiments[].power_summary.image_scoring_plugin_gpu_power_consumption` | number / null | no | — | Scoring plugin GPU watts |
| `runtime.experiments[].power_summary.total_cpu_power_consumption` | number / null | no | — | Sum of all CPU plugin watts |
| `runtime.experiments[].power_summary.total_gpu_power_consumption` | number / null | no | — | Sum of all GPU plugin watts |


MLHub

Fields available from the HuggingFace API and model card metadata.

| # | Field | Type | Notes |
|---|---|---|---|
| 1 | `name` | string | Repository name (last segment of `modelId`) |
| 2 | `model_type` | string | Architecture family (from `config.json`) |
| 3 | `version` | string | Model version tag |
| 4 | `framework` | string | ML library (e.g. `transformers`, `pytorch`) |
| 5 | `image` | string | Associated image or logo URL |
| 6 | `labels` | string[] | Inference class labels |
| 7 | `label_map` | object | Integer → label string mapping |
| 8 | `multi_modal` | boolean | Whether the model accepts multiple input modalities |
| 9 | `model_inputs` | ModelIO | Input tensor specs |
| 10 | `model_outputs` | ModelIO | Output tensor specs |
| 11 | `task_types` | string[] | HuggingFace pipeline tags (e.g. `object-detection`) |
| 12 | `inference_precision` | string | Numeric precision used at inference (e.g. `fp16`) |
| 13 | `inference_hardware` | HardwareRequirements | Recommended hardware for inference |
| 14 | `inference_software_dependencies` | string[] | Package requirements for inference |
| 15 | `inference_max_energy_consumption_watts` | float | Peak power draw during inference |
| 16 | `inference_max_latency_ms` | float | Maximum acceptable inference latency |
| 17 | `inference_min_throughput` | float | Minimum images/sec at inference |
| 18 | `inference_max_compute_utilization_percentage` | float | Peak GPU/CPU utilisation during inference |
| 19 | `inference_max_memory_usage_mb` | float | Peak memory usage during inference |
| 20 | `inference_distributed` | boolean | Whether inference can run across multiple devices |
| 21 | `training_time` | float | Total training time (hours) |
| 22 | `training_precision` | string | Numeric precision used during training |
| 23 | `training_hardware` | HardwareRequirements | Hardware used for training |
| 24 | `pretraining_datasets` | string[] | Datasets used for pretraining |
| 25 | `finetuning_datasets` | string[] | Datasets used for fine-tuning |
| 26 | `edge_optimized` | boolean | Optimised for edge deployment |
| 27 | `quantization_aware` | boolean | Trained with quantization-aware training |
| 28 | `supports_quantization` | boolean | Can be post-training quantized |
| 29 | `pretrained` | boolean | Released as a pretrained checkpoint |
| 30 | `pruned` | boolean | Weights have been pruned |
| 31 | `slimmed` | boolean | Model has been slimmed / distilled |
| 32 | `training_distributed` | boolean | Trained across multiple devices |
| 33 | `training_max_energy_consumption_watts` | float | Peak power draw during training |
| 34 | `regulatory` | string[] | Regulatory or compliance tags |
| 35 | `license` | string | SPDX license identifier |
| 36 | `bias_evaluation_score` | float | Bias evaluation score (0–1) |

**36 fields · coverage against Patra: 27.78%**

```json
{
  "name": "MegaDetector",
  "model_type": "yolov5",
  "version": "5.0",
  "framework": "pytorch",
  "image": null,
  "labels": ["animal", "human", "vehicle"],
  "label_map": { "1": "animal", "2": "human", "3": "vehicle" },
  "multi_modal": false,
  "model_inputs": { "name": "image", "shape": [1, 3, 1280, 1280], "dtype": "float32" },
  "model_outputs": { "name": "detections", "format": "xyxy + confidence + class" },
  "task_types": ["object-detection"],
  "inference_precision": "fp32",
  "inference_hardware": { "gpu": true, "min_vram_mb": 4096 },
  "inference_software_dependencies": ["torch==2.1.0", "torchvision==0.16.0", "yolov5==7.0"],
  "inference_max_energy_consumption_watts": 25.0,
  "inference_max_latency_ms": 120.0,
  "inference_min_throughput": 8.0,
  "inference_max_compute_utilization_percentage": 85.0,
  "inference_max_memory_usage_mb": 3800.0,
  "inference_distributed": false,
  "training_time": 168.0,
  "training_precision": "fp16",
  "training_hardware": { "gpu": true, "type": "A100", "count": 8 },
  "pretraining_datasets": ["https://lila.science/datasets/camera-traps-community-datasets"],
  "finetuning_datasets": [],
  "edge_optimized": true,
  "quantization_aware": false,
  "supports_quantization": true,
  "pretrained": true,
  "pruned": false,
  "slimmed": false,
  "training_distributed": true,
  "training_max_energy_consumption_watts": 3200.0,
  "regulatory": [],
  "license": "MIT",
  "bias_evaluation_score": 0.91
}
```