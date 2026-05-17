#!/usr/bin/env python3
"""Generate 20 synthetic HuggingFace API-shaped model cards for the augmentation POC.

Each card mimics the JSON returned by GET /api/models/{repo_id}, modeled on
real ICICLE-AI cards (poc/input_cards/00-03). Cards have natural sparsity —
some fields are missing depending on the model type, just like real HuggingFace.

Usage:
    python poc/generate_synthetic_dataset.py
    # writes poc/synthetic_hf_cards.json + poc/input_cards/04-23_*.json
"""

import json
import uuid
from pathlib import Path

# ---------------------------------------------------------------------------
# 20 synthetic HF API cards across 6 domains
# Each has a "full" ground truth and a "sparse" version (the actual input).
# Sparsity is natural: some models don't have pipeline_tag, library_name, etc.
# ---------------------------------------------------------------------------

CARDS = [
    # === Computer Vision (4) ===
    {
        "sparse": {
            "_id": uuid.uuid4().hex[:24],
            "id": "microsoft/resnet-50",
            "modelId": "microsoft/resnet-50",
            "author": "microsoft",
            "private": False,
            "gated": False,
            "disabled": False,
            "pipeline_tag": "image-classification",
            "library_name": "transformers",
            "tags": ["transformers", "pytorch", "safetensors", "resnet", "image-classification",
                     "vision", "dataset:imagenet-1k", "arxiv:1512.03385", "license:apache-2.0", "region:us"],
            "downloads": 313630,
            "likes": 492,
            "lastModified": "2024-02-13T21:24:05.000Z",
            "createdAt": "2022-03-16T15:42:43.000Z",
            "model-index": None,
            "config": {"architectures": ["ResNetForImageClassification"], "model_type": "resnet"},
            "cardData": {"license": "apache-2.0", "tags": ["vision", "image-classification"], "datasets": ["imagenet-1k"]},
            "transformersInfo": {"auto_model": "AutoModelForImageClassification", "pipeline_tag": "image-classification"},
            "siblings": [{"rfilename": "README.md"}, {"rfilename": "config.json"}, {"rfilename": "model.safetensors"}],
            "spaces": [],
            "usedStorage": 410255200,
        },
        "ground_truth": {
            "name": "resnet-50",
            "short_description": "ResNet-50 model pre-trained on ImageNet-1k for image classification at 224x224 resolution.",
            "full_description": "ResNet-50 is a 50-layer deep residual network pre-trained on ImageNet-1k. Introduced in Deep Residual Learning for Image Recognition, it achieves 76.1% top-1 accuracy. Widely used as a backbone for transfer learning in computer vision tasks.",
            "keywords": "vision, image-classification",
            "author": "microsoft",
            "citation": None,
            "input_data": "https://huggingface.co/datasets/imagenet-1k",
            "input_type": "Image",
            "foundational_model": "resnet-50",
            "category": "classification",
            "ai_model": {"name": "resnet-50", "framework": "pytorch", "license": "apache-2.0", "model_type": "cnn"},
        },
        "domain": "cv",
    },
    {
        "sparse": {
            "_id": uuid.uuid4().hex[:24],
            "id": "ultralytics/yolov8n",
            "modelId": "ultralytics/yolov8n",
            "author": "ultralytics",
            "private": False,
            "gated": False,
            "disabled": False,
            "pipeline_tag": "object-detection",
            "library_name": "ultralytics",
            "tags": ["ultralytics", "object-detection", "yolo", "yolov8", "real-time",
                     "dataset:coco", "license:agpl-3.0", "region:us"],
            "downloads": 85000,
            "likes": 310,
            "lastModified": "2024-06-01T10:00:00.000Z",
            "createdAt": "2023-01-10T08:00:00.000Z",
            "model-index": None,
            "config": {},
            "cardData": {"license": "agpl-3.0", "tags": ["object-detection", "yolo", "yolov8", "real-time"], "datasets": ["coco"]},
            "siblings": [{"rfilename": "README.md"}, {"rfilename": "yolov8n.pt"}, {"rfilename": "args.yaml"}],
            "spaces": [],
            "usedStorage": 12400000,
        },
        "ground_truth": {
            "name": "yolov8n",
            "short_description": "YOLOv8 Nano model for real-time object detection on COCO.",
            "full_description": "YOLOv8 Nano is an ultralight real-time object detection model trained on the COCO dataset at 640x640 input resolution. It achieves 37.3 mAP on COCO val2017 at 80 FPS on NVIDIA T4. Ideal for edge deployment and real-time inference.",
            "keywords": "object-detection, yolo, yolov8, real-time",
            "author": "ultralytics",
            "citation": None,
            "input_data": "https://huggingface.co/datasets/coco",
            "input_type": "Image",
            "foundational_model": None,
            "category": "computer vision",
            "ai_model": {"name": "yolov8n", "framework": "pytorch", "license": "agpl-3.0", "model_type": "other"},
        },
        "domain": "cv",
    },
    {
        "sparse": {
            "_id": uuid.uuid4().hex[:24],
            "id": "google/vit-base-patch16-224",
            "modelId": "google/vit-base-patch16-224",
            "author": "google",
            "private": False,
            "gated": False,
            "disabled": False,
            "pipeline_tag": "image-classification",
            "library_name": "transformers",
            "tags": ["transformers", "pytorch", "jax", "safetensors", "vit", "image-classification",
                     "dataset:imagenet-1k", "dataset:imagenet-21k", "arxiv:2010.11929", "license:apache-2.0", "region:us"],
            "downloads": 520000,
            "likes": 850,
            "lastModified": "2024-03-20T14:00:00.000Z",
            "createdAt": "2021-08-15T12:00:00.000Z",
            "model-index": None,
            "config": {"architectures": ["ViTForImageClassification"], "model_type": "vit"},
            "cardData": {"license": "apache-2.0", "tags": ["vision", "image-classification"], "datasets": ["imagenet-1k", "imagenet-21k"]},
            "transformersInfo": {"auto_model": "AutoModelForImageClassification", "pipeline_tag": "image-classification"},
            "siblings": [{"rfilename": "README.md"}, {"rfilename": "config.json"}, {"rfilename": "model.safetensors"}, {"rfilename": "preprocessor_config.json"}],
            "spaces": [],
            "usedStorage": 350000000,
        },
        "ground_truth": {
            "name": "vit-base-patch16-224",
            "short_description": "Vision Transformer (ViT) base model pre-trained on ImageNet-21k and fine-tuned on ImageNet-1k.",
            "full_description": "ViT-Base with 16x16 patch size, pre-trained on ImageNet-21k and fine-tuned on ImageNet-1k at 224x224 resolution. Achieves 84.5% top-1 accuracy. Uses self-attention over image patches instead of convolutions.",
            "keywords": "vision, image-classification",
            "author": "google",
            "citation": None,
            "input_data": "https://huggingface.co/datasets/imagenet-1k, https://huggingface.co/datasets/imagenet-21k",
            "input_type": "Image",
            "foundational_model": "vit-base-patch16-224",
            "category": "classification",
            "ai_model": {"name": "vit-base-patch16-224", "framework": "pytorch", "license": "apache-2.0", "model_type": "cnn"},
        },
        "domain": "cv",
    },
    {
        # Sparse CV card — no config, no transformersInfo (like ICICLE YOLOv9)
        "sparse": {
            "_id": uuid.uuid4().hex[:24],
            "id": "ICICLE-AI/wildlife-detector-v2",
            "modelId": "ICICLE-AI/wildlife-detector-v2",
            "author": "ICICLE-AI",
            "private": False,
            "gated": False,
            "disabled": False,
            "pipeline_tag": "object-detection",
            "library_name": "ultralytics",
            "tags": ["ultralytics", "object-detection", "yolo", "wildlife", "ecology",
                     "dataset:ICICLE-AI/wildlife-camera-traps", "license:mit", "region:us"],
            "downloads": 12,
            "likes": 3,
            "lastModified": "2025-09-15T08:30:00.000Z",
            "createdAt": "2025-08-20T14:00:00.000Z",
            "model-index": None,
            "config": {},
            "cardData": {"license": "mit", "tags": ["object-detection", "wildlife", "ecology"], "datasets": ["ICICLE-AI/wildlife-camera-traps"]},
            "siblings": [{"rfilename": "README.md"}, {"rfilename": "best.pt"}, {"rfilename": "args.yaml"}],
            "spaces": [],
            "usedStorage": 25000000,
        },
        "ground_truth": {
            "name": "wildlife-detector-v2",
            "short_description": "YOLO-based wildlife detector trained on camera trap imagery for ecological monitoring.",
            "full_description": "Wildlife detector v2 is a YOLO object detection model trained on the ICICLE wildlife camera trap dataset. Designed for automated species identification in ecological field studies. Detects and classifies common North American wildlife species.",
            "keywords": "object-detection, wildlife, ecology",
            "author": "ICICLE-AI",
            "citation": None,
            "input_data": "https://huggingface.co/datasets/ICICLE-AI/wildlife-camera-traps",
            "input_type": "Image",
            "foundational_model": None,
            "category": "computer vision",
            "ai_model": {"name": "wildlife-detector-v2", "framework": "pytorch", "license": "mit", "model_type": "other"},
        },
        "domain": "cv",
    },

    # === NLP (4) ===
    {
        "sparse": {
            "_id": uuid.uuid4().hex[:24],
            "id": "google-bert/bert-base-uncased",
            "modelId": "google-bert/bert-base-uncased",
            "author": "google-bert",
            "private": False,
            "gated": False,
            "disabled": False,
            "pipeline_tag": "fill-mask",
            "library_name": "transformers",
            "tags": ["transformers", "pytorch", "tf", "jax", "safetensors", "bert", "fill-mask",
                     "en", "dataset:bookcorpus", "dataset:wikipedia", "arxiv:1810.04805", "license:apache-2.0", "region:us"],
            "downloads": 7800000,
            "likes": 1200,
            "lastModified": "2024-01-10T08:00:00.000Z",
            "createdAt": "2019-09-12T10:00:00.000Z",
            "model-index": None,
            "config": {"architectures": ["BertForMaskedLM"], "model_type": "bert"},
            "cardData": {"license": "apache-2.0", "tags": ["bert", "fill-mask", "english"], "datasets": ["bookcorpus", "wikipedia"]},
            "transformersInfo": {"auto_model": "AutoModelForMaskedLM", "pipeline_tag": "fill-mask"},
            "siblings": [{"rfilename": "README.md"}, {"rfilename": "config.json"}, {"rfilename": "model.safetensors"}, {"rfilename": "tokenizer.json"}, {"rfilename": "vocab.txt"}],
            "spaces": [],
            "usedStorage": 440000000,
        },
        "ground_truth": {
            "name": "bert-base-uncased",
            "short_description": "BERT base model (uncased) pre-trained on English text for masked language modeling.",
            "full_description": "BERT base uncased model with 12 layers, 768 hidden size, 12 attention heads, and 110M parameters. Pre-trained on BookCorpus and English Wikipedia. Suitable for fine-tuning on downstream NLP tasks including classification, QA, and NER.",
            "keywords": "bert, fill-mask, english",
            "author": "google-bert",
            "citation": None,
            "input_data": "https://huggingface.co/datasets/bookcorpus, https://huggingface.co/datasets/wikipedia",
            "input_type": "Text",
            "foundational_model": None,
            "category": "natural language processing",
            "ai_model": {"name": "bert-base-uncased", "framework": "pytorch", "license": "apache-2.0", "model_type": "dnn"},
        },
        "domain": "nlp",
    },
    {
        "sparse": {
            "_id": uuid.uuid4().hex[:24],
            "id": "meta-llama/Llama-3.1-8B-Instruct",
            "modelId": "meta-llama/Llama-3.1-8B-Instruct",
            "author": "meta-llama",
            "private": False,
            "gated": True,
            "disabled": False,
            "pipeline_tag": "text-generation",
            "library_name": "transformers",
            "tags": ["transformers", "pytorch", "safetensors", "llama", "text-generation",
                     "conversational", "en", "de", "fr", "it", "pt", "hi", "es", "th",
                     "arxiv:2407.21783", "license:llama3.1", "region:us"],
            "downloads": 4200000,
            "likes": 3500,
            "lastModified": "2024-10-01T08:00:00.000Z",
            "createdAt": "2024-07-23T00:00:00.000Z",
            "model-index": None,
            "config": {"architectures": ["LlamaForCausalLM"], "model_type": "llama"},
            "cardData": {"license": "llama3.1", "tags": ["llama", "text-generation", "conversational"], "language": ["en", "de", "fr", "it", "pt", "hi", "es", "th"]},
            "transformersInfo": {"auto_model": "AutoModelForCausalLM", "pipeline_tag": "text-generation"},
            "siblings": [{"rfilename": "README.md"}, {"rfilename": "config.json"}, {"rfilename": "model-00001-of-00004.safetensors"}, {"rfilename": "tokenizer.json"}],
            "spaces": [],
            "usedStorage": 16000000000,
        },
        "ground_truth": {
            "name": "Llama-3.1-8B-Instruct",
            "short_description": "Llama 3.1 8B instruction-tuned model for dialogue and text generation.",
            "full_description": "Llama 3.1 8B Instruct is optimized for dialogue and instruction following with 128K context length. Trained with RLHF on instruction-tuning datasets. Supports 8 languages including English, German, French, Italian, Portuguese, Hindi, Spanish, and Thai.",
            "keywords": "llama, text-generation, conversational",
            "author": "meta-llama",
            "citation": None,
            "input_data": None,
            "input_type": "Text",
            "foundational_model": None,
            "category": "natural language processing",
            "ai_model": {"name": "Llama-3.1-8B-Instruct", "framework": "pytorch", "license": "llama3.1", "model_type": "llm"},
        },
        "domain": "nlp",
    },
    {
        "sparse": {
            "_id": uuid.uuid4().hex[:24],
            "id": "google-t5/t5-small",
            "modelId": "google-t5/t5-small",
            "author": "google-t5",
            "private": False,
            "gated": False,
            "disabled": False,
            "pipeline_tag": "text2text-generation",
            "library_name": "transformers",
            "tags": ["transformers", "pytorch", "tf", "jax", "safetensors", "t5", "text2text-generation",
                     "en", "dataset:c4", "arxiv:1910.10683", "license:apache-2.0", "region:us"],
            "downloads": 2100000,
            "likes": 420,
            "lastModified": "2024-02-01T08:00:00.000Z",
            "createdAt": "2019-11-15T10:00:00.000Z",
            "model-index": None,
            "config": {"architectures": ["T5ForConditionalGeneration"], "model_type": "t5"},
            "cardData": {"license": "apache-2.0", "tags": ["t5", "seq2seq"], "datasets": ["c4"]},
            "transformersInfo": {"auto_model": "AutoModelForSeq2SeqLM", "pipeline_tag": "text2text-generation"},
            "siblings": [{"rfilename": "README.md"}, {"rfilename": "config.json"}, {"rfilename": "model.safetensors"}, {"rfilename": "tokenizer.json"}],
            "spaces": [],
            "usedStorage": 242000000,
        },
        "ground_truth": {
            "name": "t5-small",
            "short_description": "T5 Small model for text-to-text generation tasks.",
            "full_description": "T5 Small with 60M parameters, pre-trained on the C4 corpus using a text-to-text framework. Handles translation, summarization, question answering, and classification by casting all tasks as text generation.",
            "keywords": "t5, seq2seq",
            "author": "google-t5",
            "citation": None,
            "input_data": "https://huggingface.co/datasets/c4",
            "input_type": "Text",
            "foundational_model": None,
            "category": "natural language processing",
            "ai_model": {"name": "t5-small", "framework": "pytorch", "license": "apache-2.0", "model_type": "dnn"},
        },
        "domain": "nlp",
    },
    {
        # Sparse NLP — no datasets, no config (like ICICLE GNN sparsity)
        "sparse": {
            "_id": uuid.uuid4().hex[:24],
            "id": "ICICLE-AI/policy-ner-agriculture",
            "modelId": "ICICLE-AI/policy-ner-agriculture",
            "author": "ICICLE-AI",
            "private": False,
            "gated": "auto",
            "disabled": False,
            "pipeline_tag": "token-classification",
            "tags": ["token-classification", "NER", "agriculture", "policy", "pytorch",
                     "license:mit", "region:us"],
            "downloads": 5,
            "likes": 1,
            "lastModified": "2025-11-01T12:00:00.000Z",
            "createdAt": "2025-10-20T09:00:00.000Z",
            "model-index": None,
            "config": {},
            "cardData": {"license": "mit", "tags": ["NER", "agriculture", "policy"]},
            "siblings": [{"rfilename": "README.md"}, {"rfilename": "model.safetensors"}, {"rfilename": "config.json"}, {"rfilename": "tokenizer.json"}],
            "spaces": [],
            "usedStorage": 450000000,
        },
        "ground_truth": {
            "name": "policy-ner-agriculture",
            "short_description": "Named entity recognition model for extracting policy terms from agricultural documents.",
            "full_description": "Fine-tuned NER model for identifying policy-relevant entities in agricultural text documents. Extracts entities such as crop names, regulations, geographic regions, and policy instruments from USDA and state-level agricultural policy documents.",
            "keywords": "NER, agriculture, policy",
            "author": "ICICLE-AI",
            "citation": None,
            "input_data": None,
            "input_type": "Text",
            "foundational_model": None,
            "category": "classification",
            "ai_model": {"name": "policy-ner-agriculture", "framework": "other", "license": "mit", "model_type": "other"},
        },
        "domain": "nlp",
    },

    # === Audio (3) ===
    {
        "sparse": {
            "_id": uuid.uuid4().hex[:24],
            "id": "openai/whisper-large-v3",
            "modelId": "openai/whisper-large-v3",
            "author": "openai",
            "private": False,
            "gated": False,
            "disabled": False,
            "pipeline_tag": "automatic-speech-recognition",
            "library_name": "transformers",
            "tags": ["transformers", "pytorch", "safetensors", "whisper", "automatic-speech-recognition",
                     "multilingual", "arxiv:2212.04356", "license:apache-2.0", "region:us"],
            "downloads": 9500000,
            "likes": 4200,
            "lastModified": "2024-04-15T08:00:00.000Z",
            "createdAt": "2023-11-06T00:00:00.000Z",
            "model-index": None,
            "config": {"architectures": ["WhisperForConditionalGeneration"], "model_type": "whisper"},
            "cardData": {"license": "apache-2.0", "tags": ["whisper", "asr", "multilingual"]},
            "transformersInfo": {"auto_model": "AutoModelForSpeechSeq2Seq", "pipeline_tag": "automatic-speech-recognition"},
            "siblings": [{"rfilename": "README.md"}, {"rfilename": "config.json"}, {"rfilename": "model.safetensors"}, {"rfilename": "preprocessor_config.json"}],
            "spaces": [],
            "usedStorage": 3090000000,
        },
        "ground_truth": {
            "name": "whisper-large-v3",
            "short_description": "Whisper Large V3 model for multilingual automatic speech recognition.",
            "full_description": "Whisper Large V3 with 1.5B parameters for automatic speech recognition. Supports 99 languages. Trained on 5 million hours of weakly supervised audio data. Third iteration of the Whisper large model with improved accuracy.",
            "keywords": "whisper, asr, multilingual",
            "author": "openai",
            "citation": None,
            "input_data": None,
            "input_type": "Audio",
            "foundational_model": None,
            "category": "classification",
            "ai_model": {"name": "whisper-large-v3", "framework": "pytorch", "license": "apache-2.0", "model_type": "dnn"},
        },
        "domain": "audio",
    },
    {
        "sparse": {
            "_id": uuid.uuid4().hex[:24],
            "id": "suno/bark",
            "modelId": "suno/bark",
            "author": "suno",
            "private": False,
            "gated": False,
            "disabled": False,
            "pipeline_tag": "text-to-audio",
            "library_name": "transformers",
            "tags": ["transformers", "pytorch", "bark", "text-to-audio", "text-to-speech",
                     "license:mit", "region:us"],
            "downloads": 180000,
            "likes": 950,
            "lastModified": "2024-01-20T08:00:00.000Z",
            "createdAt": "2023-04-15T00:00:00.000Z",
            "model-index": None,
            "config": {},
            "cardData": {"license": "mit", "tags": ["bark", "tts", "text-to-speech", "audio-generation"]},
            "siblings": [{"rfilename": "README.md"}, {"rfilename": "config.json"}, {"rfilename": "pytorch_model.bin"}],
            "spaces": [],
            "usedStorage": 5500000000,
        },
        "ground_truth": {
            "name": "bark",
            "short_description": "Transformer-based text-to-audio model for generating speech, music, and sound effects.",
            "full_description": "Bark is a transformer-based text-to-audio model that generates realistic speech, music, and sound effects. Supports multiple languages and speakers with voice cloning capability. Can produce non-verbal sounds like laughter and sighs.",
            "keywords": "bark, tts, text-to-speech, audio-generation",
            "author": "suno",
            "citation": None,
            "input_data": None,
            "input_type": "Text",
            "foundational_model": None,
            "category": "generative modeling",
            "ai_model": {"name": "bark", "framework": "pytorch", "license": "mit", "model_type": "other"},
        },
        "domain": "audio",
    },
    {
        # Sparse audio — no library_name (like ICICLE GNN)
        "sparse": {
            "_id": uuid.uuid4().hex[:24],
            "id": "ICICLE-AI/birdsong-classifier-v1",
            "modelId": "ICICLE-AI/birdsong-classifier-v1",
            "author": "ICICLE-AI",
            "private": False,
            "gated": False,
            "disabled": False,
            "pipeline_tag": "audio-classification",
            "tags": ["audio-classification", "birdsong", "ecology", "pytorch",
                     "license:mit", "region:us"],
            "downloads": 8,
            "likes": 2,
            "lastModified": "2025-07-10T14:00:00.000Z",
            "createdAt": "2025-06-01T09:00:00.000Z",
            "model-index": None,
            "config": {},
            "cardData": {"license": "mit", "tags": ["audio-classification", "birdsong", "ecology"]},
            "siblings": [{"rfilename": "README.md"}, {"rfilename": "model.pt"}, {"rfilename": "labels.json"}],
            "spaces": [],
            "usedStorage": 85000000,
        },
        "ground_truth": {
            "name": "birdsong-classifier-v1",
            "short_description": "Audio classifier for identifying bird species from field recordings.",
            "full_description": "Birdsong classifier trained on field recordings for automated species identification. Supports common North American bird species. Designed for ecological survey automation and biodiversity monitoring in ICICLE smart agriculture contexts.",
            "keywords": "audio-classification, birdsong, ecology",
            "author": "ICICLE-AI",
            "citation": None,
            "input_data": None,
            "input_type": "Audio",
            "foundational_model": None,
            "category": "classification",
            "ai_model": {"name": "birdsong-classifier-v1", "framework": "other", "license": "mit", "model_type": "other"},
        },
        "domain": "audio",
    },

    # === Tabular / Classical ML (3) ===
    {
        # Sparse — no pipeline_tag, no config (like ICICLE NAS)
        "sparse": {
            "_id": uuid.uuid4().hex[:24],
            "id": "scikit-learn-mojos/randomforest-iris",
            "modelId": "scikit-learn-mojos/randomforest-iris",
            "author": "scikit-learn-mojos",
            "private": False,
            "gated": False,
            "disabled": False,
            "tags": ["scikit-learn", "tabular-classification", "classification", "iris", "random-forest",
                     "license:bsd-3-clause", "region:us"],
            "downloads": 450,
            "likes": 12,
            "lastModified": "2024-05-01T08:00:00.000Z",
            "createdAt": "2023-12-01T10:00:00.000Z",
            "model-index": None,
            "config": {},
            "cardData": {"license": "bsd-3-clause", "tags": ["tabular-classification", "iris", "random-forest"]},
            "siblings": [{"rfilename": "README.md"}, {"rfilename": "model.pkl"}, {"rfilename": "requirements.txt"}],
            "spaces": [],
            "usedStorage": 2500000,
        },
        "ground_truth": {
            "name": "randomforest-iris",
            "short_description": "Random Forest classifier for Iris species classification.",
            "full_description": "Random Forest classifier trained on the Iris dataset for species classification. Uses 100 estimators with max_depth=5. Achieves 97% accuracy on the test set. Serialized as a scikit-learn pickle.",
            "keywords": "tabular-classification, iris, random-forest",
            "author": "scikit-learn-mojos",
            "citation": None,
            "input_data": None,
            "input_type": "Tabular",
            "foundational_model": None,
            "category": "classification",
            "ai_model": {"name": "randomforest-iris", "framework": "other", "license": "bsd-3-clause", "model_type": "random_forest"},
        },
        "domain": "tabular",
    },
    {
        "sparse": {
            "_id": uuid.uuid4().hex[:24],
            "id": "xgboost-community/xgb-housing-regressor",
            "modelId": "xgboost-community/xgb-housing-regressor",
            "author": "xgboost-community",
            "private": False,
            "gated": False,
            "disabled": False,
            "tags": ["xgboost", "tabular-regression", "regression", "housing", "gradient-boosting",
                     "dataset:california_housing", "license:apache-2.0", "region:us"],
            "downloads": 200,
            "likes": 5,
            "lastModified": "2024-03-10T08:00:00.000Z",
            "createdAt": "2024-01-05T10:00:00.000Z",
            "model-index": None,
            "config": {},
            "cardData": {"license": "apache-2.0", "tags": ["regression", "housing", "gradient-boosting"], "datasets": ["california_housing"]},
            "siblings": [{"rfilename": "README.md"}, {"rfilename": "model.json"}, {"rfilename": "requirements.txt"}],
            "spaces": [],
            "usedStorage": 1200000,
        },
        "ground_truth": {
            "name": "xgb-housing-regressor",
            "short_description": "XGBoost gradient boosting regressor for California housing price prediction.",
            "full_description": "XGBoost regressor trained on the California Housing dataset. Predicts median house values using 200 estimators with learning rate 0.1. Achieves RMSE of 0.46 on the test set.",
            "keywords": "regression, housing, gradient-boosting",
            "author": "xgboost-community",
            "citation": None,
            "input_data": "https://huggingface.co/datasets/california_housing",
            "input_type": "Tabular",
            "foundational_model": None,
            "category": "regression",
            "ai_model": {"name": "xgb-housing-regressor", "framework": "other", "license": "apache-2.0", "model_type": "other"},
        },
        "domain": "tabular",
    },
    {
        # Minimal tabular — almost nothing (like ICICLE NAS)
        "sparse": {
            "_id": uuid.uuid4().hex[:24],
            "id": "ICICLE-AI/crop-yield-predictor",
            "modelId": "ICICLE-AI/crop-yield-predictor",
            "author": "ICICLE-AI",
            "private": False,
            "gated": False,
            "disabled": False,
            "tags": ["license:mit", "region:us"],
            "downloads": 0,
            "likes": 0,
            "lastModified": "2026-02-01T10:00:00.000Z",
            "createdAt": "2026-01-15T08:00:00.000Z",
            "model-index": None,
            "config": {},
            "cardData": {"license": "mit"},
            "siblings": [{"rfilename": "README.md"}, {"rfilename": "model.pkl"}, {"rfilename": "train.py"}, {"rfilename": "data/county_yields.csv"}],
            "spaces": [],
            "usedStorage": 3500000,
        },
        "ground_truth": {
            "name": "crop-yield-predictor",
            "short_description": "Crop yield prediction model for US county-level agricultural forecasting.",
            "full_description": "Machine learning model for predicting crop yields at US county level. Uses historical yield data, weather features, and soil characteristics. Part of the ICICLE smart agriculture initiative for food security planning.",
            "keywords": "agriculture, crop-yield, prediction",
            "author": "ICICLE-AI",
            "citation": None,
            "input_data": None,
            "input_type": "Tabular",
            "foundational_model": None,
            "category": "other",
            "ai_model": {"name": "crop-yield-predictor", "framework": "other", "license": "mit", "model_type": "other"},
        },
        "domain": "tabular",
    },

    # === Multimodal (3) ===
    {
        "sparse": {
            "_id": uuid.uuid4().hex[:24],
            "id": "openai/clip-vit-base-patch32",
            "modelId": "openai/clip-vit-base-patch32",
            "author": "openai",
            "private": False,
            "gated": False,
            "disabled": False,
            "pipeline_tag": "zero-shot-image-classification",
            "library_name": "transformers",
            "tags": ["transformers", "pytorch", "jax", "clip", "zero-shot-image-classification",
                     "multimodal", "arxiv:2103.00020", "license:mit", "region:us"],
            "downloads": 2800000,
            "likes": 1100,
            "lastModified": "2024-02-10T08:00:00.000Z",
            "createdAt": "2021-03-01T00:00:00.000Z",
            "model-index": None,
            "config": {"architectures": ["CLIPModel"], "model_type": "clip"},
            "cardData": {"license": "mit", "tags": ["clip", "multimodal", "vision-language", "zero-shot"]},
            "transformersInfo": {"auto_model": "AutoModel", "pipeline_tag": "zero-shot-image-classification"},
            "siblings": [{"rfilename": "README.md"}, {"rfilename": "config.json"}, {"rfilename": "model.safetensors"}, {"rfilename": "preprocessor_config.json"}, {"rfilename": "tokenizer.json"}],
            "spaces": [],
            "usedStorage": 605000000,
        },
        "ground_truth": {
            "name": "clip-vit-base-patch32",
            "short_description": "CLIP model with ViT-B/32 for zero-shot image classification via text-image embeddings.",
            "full_description": "CLIP (Contrastive Language-Image Pre-training) with ViT-B/32 visual encoder. Learns visual concepts from natural language supervision. Enables zero-shot image classification by comparing image and text embeddings without task-specific fine-tuning.",
            "keywords": "clip, multimodal, vision-language, zero-shot",
            "author": "openai",
            "citation": None,
            "input_data": None,
            "input_type": "Multimodal",
            "foundational_model": None,
            "category": "classification",
            "ai_model": {"name": "clip-vit-base-patch32", "framework": "pytorch", "license": "mit", "model_type": "dnn"},
        },
        "domain": "multimodal",
    },
    {
        "sparse": {
            "_id": uuid.uuid4().hex[:24],
            "id": "stabilityai/stable-diffusion-xl-base-1.0",
            "modelId": "stabilityai/stable-diffusion-xl-base-1.0",
            "author": "stabilityai",
            "private": False,
            "gated": False,
            "disabled": False,
            "pipeline_tag": "text-to-image",
            "library_name": "diffusers",
            "tags": ["diffusers", "safetensors", "stable-diffusion", "text-to-image", "image-generation", "sdxl",
                     "license:openrail++", "region:us"],
            "downloads": 3200000,
            "likes": 6500,
            "lastModified": "2024-05-01T08:00:00.000Z",
            "createdAt": "2023-07-26T00:00:00.000Z",
            "model-index": None,
            "config": {},
            "cardData": {"license": "openrail++", "tags": ["stable-diffusion", "text-to-image", "sdxl"]},
            "siblings": [{"rfilename": "README.md"}, {"rfilename": "model_index.json"}, {"rfilename": "unet/diffusion_pytorch_model.safetensors"}],
            "spaces": [],
            "usedStorage": 6940000000,
        },
        "ground_truth": {
            "name": "stable-diffusion-xl-base-1.0",
            "short_description": "Stable Diffusion XL Base 1.0 for high-resolution text-to-image generation.",
            "full_description": "SDXL Base 1.0 generates 1024x1024 images from text prompts. 3.5B parameter UNet with dual text encoders (CLIP ViT-L and OpenCLIP ViT-bigG). Significant improvement in image quality and prompt adherence over SD 1.5/2.1.",
            "keywords": "stable-diffusion, text-to-image, sdxl",
            "author": "stabilityai",
            "citation": None,
            "input_data": None,
            "input_type": "Text",
            "foundational_model": None,
            "category": "generative modeling",
            "ai_model": {"name": "stable-diffusion-xl-base-1.0", "framework": "pytorch", "license": "openrail++", "model_type": "other"},
        },
        "domain": "multimodal",
    },
    {
        # Sparse multimodal — no library_name, no config
        "sparse": {
            "_id": uuid.uuid4().hex[:24],
            "id": "ICICLE-AI/satellite-crop-captioner",
            "modelId": "ICICLE-AI/satellite-crop-captioner",
            "author": "ICICLE-AI",
            "private": False,
            "gated": False,
            "disabled": False,
            "pipeline_tag": "image-to-text",
            "tags": ["image-to-text", "satellite", "agriculture", "captioning", "pytorch",
                     "license:mit", "region:us"],
            "downloads": 3,
            "likes": 0,
            "lastModified": "2026-01-05T10:00:00.000Z",
            "createdAt": "2025-12-10T09:00:00.000Z",
            "model-index": None,
            "config": {},
            "cardData": {"license": "mit", "tags": ["satellite", "agriculture", "captioning"]},
            "siblings": [{"rfilename": "README.md"}, {"rfilename": "model.pt"}, {"rfilename": "config.yaml"}],
            "spaces": [],
            "usedStorage": 1200000000,
        },
        "ground_truth": {
            "name": "satellite-crop-captioner",
            "short_description": "Image captioning model for describing crop conditions in satellite imagery.",
            "full_description": "Multimodal model that generates natural language descriptions of crop conditions from satellite imagery. Trained on paired satellite images and expert agronomist annotations. Outputs descriptions of crop health, growth stage, and land use patterns.",
            "keywords": "satellite, agriculture, captioning",
            "author": "ICICLE-AI",
            "citation": None,
            "input_data": None,
            "input_type": "Image",
            "foundational_model": None,
            "category": "generative modeling",
            "ai_model": {"name": "satellite-crop-captioner", "framework": "other", "license": "mit", "model_type": "other"},
        },
        "domain": "multimodal",
    },

    # === Scientific / Simulation (3) — ICICLE-style edge cases ===
    {
        # Non-ML: agent-based model (like FoodAccessModel)
        "sparse": {
            "_id": uuid.uuid4().hex[:24],
            "id": "ICICLE-AI/water-distribution-abm",
            "modelId": "ICICLE-AI/water-distribution-abm",
            "author": "ICICLE-AI",
            "private": True,
            "gated": False,
            "disabled": False,
            "library_name": "Mesa",
            "tags": ["Mesa", "license:unknown", "region:us"],
            "downloads": 0,
            "likes": 0,
            "lastModified": "2025-11-15T10:00:00.000Z",
            "createdAt": "2025-10-01T09:00:00.000Z",
            "model-index": None,
            "config": {},
            "cardData": {"license": "unknown", "library_name": "Mesa"},
            "siblings": [{"rfilename": "README.md"}, {"rfilename": "simulation/agents.py"}, {"rfilename": "simulation/model.py"}, {"rfilename": "Dockerfile"}, {"rfilename": "pyproject.toml"}],
            "spaces": [],
            "usedStorage": 0,
        },
        "ground_truth": {
            "name": "water-distribution-abm",
            "short_description": "Agent-based model for simulating water distribution networks under stress scenarios.",
            "full_description": "Mesa-based agent-based model simulating water distribution in urban networks under drought and infrastructure failure scenarios. Part of the ICICLE smart water management initiative. Agents represent households, distribution nodes, and water sources.",
            "keywords": "agent-based-model, water, simulation",
            "author": "ICICLE-AI",
            "citation": None,
            "input_data": None,
            "input_type": "Tabular",
            "foundational_model": None,
            "category": "other",
            "ai_model": {"name": "water-distribution-abm", "framework": "other", "license": "unknown", "model_type": "other"},
        },
        "domain": "scientific",
    },
    {
        # GNN — sparse, like ICICLE FoodFlow
        "sparse": {
            "_id": uuid.uuid4().hex[:24],
            "id": "ICICLE-AI/supply-chain-gnn",
            "modelId": "ICICLE-AI/supply-chain-gnn",
            "author": "ICICLE-AI",
            "private": False,
            "gated": "auto",
            "disabled": False,
            "pipeline_tag": "graph-ml",
            "tags": ["GNN", "supply-chain", "food-security", "pytorch", "graph-ml",
                     "license:mit", "region:us"],
            "downloads": 2,
            "likes": 0,
            "lastModified": "2026-03-01T12:00:00.000Z",
            "createdAt": "2026-02-15T08:00:00.000Z",
            "model-index": None,
            "config": {},
            "cardData": {"license": "mit", "pipeline_tag": "graph-ml", "tags": ["GNN", "supply-chain", "food-security"]},
            "siblings": [{"rfilename": "README.md"}, {"rfilename": "models/gcn_model.pth"}, {"rfilename": "code/model.py"}, {"rfilename": "data/supply_chain_graph.csv"}],
            "spaces": [],
            "usedStorage": 45000000,
        },
        "ground_truth": {
            "name": "supply-chain-gnn",
            "short_description": "Graph neural network for modeling food supply chain disruptions.",
            "full_description": "GNN model for predicting supply chain disruption propagation in food distribution networks. Uses graph convolutional layers to model dependencies between supply chain nodes. Part of the ICICLE food security initiative.",
            "keywords": "GNN, supply-chain, food-security",
            "author": "ICICLE-AI",
            "citation": None,
            "input_data": None,
            "input_type": "Tabular",
            "foundational_model": None,
            "category": "graph neural networks",
            "ai_model": {"name": "supply-chain-gnn", "framework": "other", "license": "mit", "model_type": "gnn"},
        },
        "domain": "scientific",
    },
    {
        # Minimal — almost nothing (like ICICLE NAS)
        "sparse": {
            "_id": uuid.uuid4().hex[:24],
            "id": "ICICLE-AI/climate-downscaling-unet",
            "modelId": "ICICLE-AI/climate-downscaling-unet",
            "author": "ICICLE-AI",
            "private": False,
            "gated": False,
            "disabled": False,
            "tags": ["climate", "downscaling", "pytorch", "license:apache-2.0", "region:us"],
            "downloads": 0,
            "likes": 0,
            "lastModified": "2026-04-01T08:00:00.000Z",
            "createdAt": "2026-03-20T10:00:00.000Z",
            "model-index": None,
            "config": {},
            "cardData": {"license": "apache-2.0"},
            "siblings": [{"rfilename": "README.md"}, {"rfilename": "model.pt"}, {"rfilename": "train.py"}],
            "spaces": [],
            "usedStorage": 250000000,
        },
        "ground_truth": {
            "name": "climate-downscaling-unet",
            "short_description": "U-Net model for statistical downscaling of climate model outputs.",
            "full_description": "U-Net architecture for downscaling coarse climate model outputs to higher spatial resolution. Takes low-resolution temperature and precipitation grids and produces fine-grained regional projections. Part of ICICLE climate adaptation research.",
            "keywords": "climate, downscaling, unet",
            "author": "ICICLE-AI",
            "citation": None,
            "input_data": None,
            "input_type": "Image",
            "foundational_model": None,
            "category": "other",
            "ai_model": {"name": "climate-downscaling-unet", "framework": "other", "license": "apache-2.0", "model_type": "cnn"},
        },
        "domain": "scientific",
    },
]

# ---------------------------------------------------------------------------
# 10 synthetic HF Dataset API cards across 5 domains
# Mimics GET /api/datasets/{namespace}/{repo}
# ---------------------------------------------------------------------------

DATASHEETS = [
    # === CV Datasets (2) ===
    {
        "sparse": {
            "_id": uuid.uuid4().hex[:24],
            "id": "ICICLE-AI/wildlife-camera-traps",
            "author": "ICICLE-AI",
            "private": False,
            "gated": False,
            "tags": ["object-detection", "ecology", "wildlife", "license:cc-by-4.0",
                     "size_categories:100K<n<1M", "task_categories:object-detection"],
            "downloads": 245,
            "likes": 18,
            "lastModified": "2025-11-15T10:00:00.000Z",
            "createdAt": "2025-08-01T09:00:00.000Z",
            "cardData": {
                "license": "cc-by-4.0",
                "task_categories": ["object-detection"],
                "pretty_name": "Wildlife Camera Trap Images",
                "size_categories": ["100K<n<1M"],
            },
            "siblings": [{"rfilename": "README.md"}, {"rfilename": "metadata.csv"}, {"rfilename": "images/"}],
        },
        "ground_truth": {
            "title": "Wildlife Camera Trap Images",
            "description": "Camera trap images of wildlife species from Midwest US conservation areas. Contains 250K JPEG images with bounding box annotations for 45 species including deer, coyote, and wild turkey. Designed for training object detection models for ecological monitoring.",
            "subjects": "object detection, wildlife, ecology, camera traps, conservation",
            "creator": "ICICLE-AI",
            "publisher": "ICICLE AI Institute",
            "resource_type": "Annotated image dataset for wildlife detection",
            "resource_type_general": "Dataset",
            "publication_year": 2025,
            "size": "250K images, 12GB",
            "format": "JPEG images + CSV metadata",
            "version": "1.0",
            "license": "cc-by-4.0",
        },
        "domain": "cv",
    },
    {
        "sparse": {
            "_id": uuid.uuid4().hex[:24],
            "id": "satellite-benchmark/eurosat",
            "author": "satellite-benchmark",
            "private": False,
            "gated": False,
            "tags": ["image-classification", "remote-sensing", "satellite", "license:mit",
                     "size_categories:10K<n<100K", "task_categories:image-classification"],
            "downloads": 3200,
            "likes": 89,
            "lastModified": "2024-06-20T14:00:00.000Z",
            "createdAt": "2023-03-10T08:00:00.000Z",
            "cardData": {
                "license": "mit",
                "task_categories": ["image-classification"],
                "pretty_name": "EuroSAT Land Use Classification",
                "size_categories": ["10K<n<100K"],
            },
            "siblings": [{"rfilename": "README.md"}, {"rfilename": "data/"}],
        },
        "ground_truth": {
            "title": "EuroSAT Land Use Classification",
            "description": "Sentinel-2 satellite images covering 13 spectral bands for land use and land cover classification across 10 classes. Contains 27,000 labeled geo-referenced images from 34 European countries.",
            "subjects": "remote sensing, satellite imagery, land use, classification, Sentinel-2",
            "creator": "satellite-benchmark",
            "publisher": "German Research Center for Artificial Intelligence (DFKI)",
            "resource_type": "Satellite image classification benchmark",
            "resource_type_general": "Dataset",
            "publication_year": 2019,
            "size": "27K images, 2.4GB",
            "format": "GeoTIFF images",
            "version": "2.0",
            "license": "mit",
        },
        "domain": "cv",
    },
    # === NLP Datasets (2) ===
    {
        "sparse": {
            "_id": uuid.uuid4().hex[:24],
            "id": "ICICLE-AI/policy-documents-qa",
            "author": "ICICLE-AI",
            "private": False,
            "gated": False,
            "tags": ["question-answering", "policy", "government", "license:cc-by-4.0",
                     "task_categories:question-answering", "language:en"],
            "downloads": 120,
            "likes": 8,
            "lastModified": "2025-09-10T16:00:00.000Z",
            "createdAt": "2025-05-20T11:00:00.000Z",
            "cardData": {
                "license": "cc-by-4.0",
                "task_categories": ["question-answering"],
                "language": ["en"],
                "pretty_name": "US Policy Documents QA",
            },
            "siblings": [{"rfilename": "README.md"}, {"rfilename": "train.jsonl"}, {"rfilename": "test.jsonl"}],
        },
        "ground_truth": {
            "title": "US Policy Documents QA",
            "description": "Question-answer pairs derived from US federal and state policy documents covering agriculture, environment, and infrastructure. Contains 15K QA pairs with source document citations for retrieval-augmented generation research.",
            "subjects": "question answering, policy documents, government, RAG, agriculture",
            "creator": "ICICLE-AI",
            "publisher": "ICICLE AI Institute",
            "resource_type": "Question-answer dataset from policy documents",
            "resource_type_general": "Dataset",
            "publication_year": 2025,
            "size": "15K QA pairs, 180MB",
            "format": "JSONL",
            "version": "1.0",
            "license": "cc-by-4.0",
        },
        "domain": "nlp",
    },
    {
        "sparse": {
            "_id": uuid.uuid4().hex[:24],
            "id": "multilingual-bench/xnli-extended",
            "author": "multilingual-bench",
            "private": False,
            "gated": False,
            "tags": ["natural-language-inference", "multilingual", "benchmark",
                     "license:apache-2.0", "task_categories:text-classification",
                     "language:en", "language:zh", "language:es", "language:ar"],
            "downloads": 5600,
            "likes": 42,
            "lastModified": "2024-12-01T09:00:00.000Z",
            "createdAt": "2024-01-15T12:00:00.000Z",
            "cardData": {
                "license": "apache-2.0",
                "task_categories": ["text-classification"],
                "language": ["en", "zh", "es", "ar", "fr", "de", "hi", "sw"],
                "pretty_name": "XNLI Extended Multilingual NLI Benchmark",
                "size_categories": ["100K<n<1M"],
            },
            "siblings": [{"rfilename": "README.md"}, {"rfilename": "data/"}],
        },
        "ground_truth": {
            "title": "XNLI Extended Multilingual NLI Benchmark",
            "description": "Extended Cross-lingual Natural Language Inference benchmark covering 8 languages. Contains 400K premise-hypothesis pairs with entailment, contradiction, and neutral labels. Includes low-resource languages (Swahili, Hindi) for evaluating multilingual transfer.",
            "subjects": "natural language inference, multilingual, benchmark, cross-lingual, NLI",
            "creator": "multilingual-bench",
            "publisher": "Multilingual NLP Benchmarks Consortium",
            "resource_type": "Multilingual NLI evaluation benchmark",
            "resource_type_general": "Dataset",
            "publication_year": 2024,
            "size": "400K pairs, 350MB",
            "format": "Parquet",
            "version": "1.2",
            "license": "apache-2.0",
        },
        "domain": "nlp",
    },
    # === Tabular Datasets (2) ===
    {
        "sparse": {
            "_id": uuid.uuid4().hex[:24],
            "id": "ICICLE-AI/midwest-crop-yields",
            "author": "ICICLE-AI",
            "private": False,
            "gated": False,
            "tags": ["tabular", "agriculture", "crop-yield", "time-series",
                     "license:cc-by-4.0", "task_categories:tabular-regression"],
            "downloads": 340,
            "likes": 25,
            "lastModified": "2025-10-01T08:00:00.000Z",
            "createdAt": "2025-03-15T10:00:00.000Z",
            "cardData": {
                "license": "cc-by-4.0",
                "task_categories": ["tabular-regression"],
                "pretty_name": "Midwest Crop Yield Records",
            },
            "siblings": [{"rfilename": "README.md"}, {"rfilename": "yields.csv"}, {"rfilename": "weather.csv"}],
        },
        "ground_truth": {
            "title": "Midwest Crop Yield Records",
            "description": "County-level crop yield data for corn, soybean, and wheat across 12 Midwest US states from 2000-2024. Includes weather features (temperature, precipitation, soil moisture) from ERA5 reanalysis. Designed for agricultural yield prediction research.",
            "subjects": "agriculture, crop yield, time series, Midwest US, weather, prediction",
            "creator": "ICICLE-AI",
            "publisher": "ICICLE AI Institute",
            "resource_type": "Tabular crop yield dataset with weather covariates",
            "resource_type_general": "Dataset",
            "publication_year": 2025,
            "size": "850K rows, 45MB",
            "format": "CSV",
            "version": "2.0",
            "license": "cc-by-4.0",
        },
        "domain": "tabular",
    },
    {
        "sparse": {
            "_id": uuid.uuid4().hex[:24],
            "id": "census-ml/acs-income-2023",
            "author": "census-ml",
            "private": False,
            "gated": False,
            "tags": ["tabular", "census", "income", "fairness",
                     "license:cc0-1.0", "task_categories:tabular-classification"],
            "downloads": 8900,
            "likes": 67,
            "lastModified": "2024-08-15T12:00:00.000Z",
            "createdAt": "2024-02-01T09:00:00.000Z",
            "cardData": {
                "license": "cc0-1.0",
                "task_categories": ["tabular-classification"],
                "pretty_name": "ACS Income Prediction 2023",
                "size_categories": ["1M<n<10M"],
            },
            "siblings": [{"rfilename": "README.md"}, {"rfilename": "data.parquet"}],
        },
        "ground_truth": {
            "title": "ACS Income Prediction 2023",
            "description": "American Community Survey microdata for income prediction, derived from 2023 PUMS files. Contains 3.2M individual records with demographic, employment, and geographic features. Includes sensitive attributes for fairness evaluation.",
            "subjects": "income prediction, census, fairness, demographics, classification",
            "creator": "census-ml",
            "publisher": "US Census Bureau (derived)",
            "resource_type": "Tabular classification benchmark with fairness attributes",
            "resource_type_general": "Dataset",
            "publication_year": 2024,
            "size": "3.2M rows, 1.1GB",
            "format": "Parquet",
            "version": "1.0",
            "license": "cc0-1.0",
        },
        "domain": "tabular",
    },
    # === Audio Datasets (2) ===
    {
        "sparse": {
            "_id": uuid.uuid4().hex[:24],
            "id": "ICICLE-AI/birdsong-midwest",
            "author": "ICICLE-AI",
            "private": False,
            "gated": False,
            "tags": ["audio-classification", "ecology", "birdsong", "bioacoustics",
                     "license:cc-by-4.0", "task_categories:audio-classification"],
            "downloads": 180,
            "likes": 14,
            "lastModified": "2025-07-20T11:00:00.000Z",
            "createdAt": "2025-04-10T08:00:00.000Z",
            "cardData": {
                "license": "cc-by-4.0",
                "task_categories": ["audio-classification"],
                "pretty_name": "Midwest Birdsong Recordings",
            },
            "siblings": [{"rfilename": "README.md"}, {"rfilename": "audio/"}, {"rfilename": "metadata.csv"}],
        },
        "ground_truth": {
            "title": "Midwest Birdsong Recordings",
            "description": "Field recordings of 120 bird species from Midwest US habitats captured using autonomous recording units. Contains 85K audio clips (5-30 seconds each) with species labels verified by expert ornithologists. Supports bioacoustic monitoring and species identification research.",
            "subjects": "bioacoustics, birdsong, species identification, ecology, audio classification",
            "creator": "ICICLE-AI",
            "publisher": "ICICLE AI Institute",
            "resource_type": "Audio classification dataset for bird species identification",
            "resource_type_general": "Dataset",
            "publication_year": 2025,
            "size": "85K clips, 48GB",
            "format": "WAV audio + CSV metadata",
            "version": "1.0",
            "license": "cc-by-4.0",
        },
        "domain": "audio",
    },
    {
        "sparse": {
            "_id": uuid.uuid4().hex[:24],
            "id": "speech-research/common-voice-18",
            "author": "speech-research",
            "private": False,
            "gated": False,
            "tags": ["automatic-speech-recognition", "multilingual", "speech",
                     "license:cc0-1.0", "task_categories:automatic-speech-recognition",
                     "language:en", "language:de", "language:fr", "language:es"],
            "downloads": 15000,
            "likes": 230,
            "lastModified": "2025-01-10T10:00:00.000Z",
            "createdAt": "2024-11-01T08:00:00.000Z",
            "cardData": {
                "license": "cc0-1.0",
                "task_categories": ["automatic-speech-recognition"],
                "language": ["en", "de", "fr", "es", "zh", "ja", "ar", "pt"],
                "pretty_name": "Common Voice Corpus 18.0",
                "size_categories": ["10M<n<100M"],
            },
            "siblings": [{"rfilename": "README.md"}, {"rfilename": "data/"}],
        },
        "ground_truth": {
            "title": "Common Voice Corpus 18.0",
            "description": "Crowdsourced multilingual speech corpus with validated transcriptions in 120+ languages. Version 18.0 contains 30K hours of validated speech from volunteer contributors worldwide. The largest open-source speech dataset available.",
            "subjects": "speech recognition, multilingual, crowdsourced, ASR, transcription",
            "creator": "speech-research",
            "publisher": "Mozilla Foundation",
            "resource_type": "Multilingual speech recognition corpus",
            "resource_type_general": "Dataset",
            "publication_year": 2024,
            "size": "30K hours, 2.5TB",
            "format": "MP3 audio + TSV metadata",
            "version": "18.0",
            "license": "cc0-1.0",
        },
        "domain": "audio",
    },
    # === Scientific Datasets (2) ===
    {
        "sparse": {
            "_id": uuid.uuid4().hex[:24],
            "id": "ICICLE-AI/era5-midwest-climate",
            "author": "ICICLE-AI",
            "private": False,
            "gated": False,
            "tags": ["climate", "reanalysis", "time-series", "geospatial",
                     "license:cc-by-4.0"],
            "downloads": 95,
            "likes": 7,
            "lastModified": "2025-06-01T09:00:00.000Z",
            "createdAt": "2025-02-15T10:00:00.000Z",
            "cardData": {
                "license": "cc-by-4.0",
                "pretty_name": "ERA5 Midwest Climate Reanalysis Subset",
            },
            "siblings": [{"rfilename": "README.md"}, {"rfilename": "data.zarr/"}],
        },
        "ground_truth": {
            "title": "ERA5 Midwest Climate Reanalysis Subset",
            "description": "Subset of ECMWF ERA5 reanalysis data covering the Midwest US (36-49N, 80-104W) at hourly resolution from 2000-2024. Includes temperature, precipitation, wind, humidity, and solar radiation variables at 0.25-degree grid spacing. Prepared for agricultural and ecological modeling.",
            "subjects": "climate, ERA5, reanalysis, Midwest US, agriculture, geospatial",
            "creator": "ICICLE-AI",
            "publisher": "ICICLE AI Institute",
            "resource_type": "Climate reanalysis subset for regional modeling",
            "resource_type_general": "Dataset",
            "publication_year": 2025,
            "size": "25 years hourly, 180GB",
            "format": "Zarr",
            "version": "1.0",
            "license": "cc-by-4.0",
        },
        "domain": "scientific",
    },
    {
        "sparse": {
            "_id": uuid.uuid4().hex[:24],
            "id": "bio-bench/protein-structure-pred",
            "author": "bio-bench",
            "private": False,
            "gated": False,
            "tags": ["biology", "protein", "structure-prediction", "benchmark",
                     "license:apache-2.0"],
            "downloads": 4200,
            "likes": 55,
            "lastModified": "2024-09-15T14:00:00.000Z",
            "createdAt": "2024-04-01T08:00:00.000Z",
            "cardData": {
                "license": "apache-2.0",
                "pretty_name": "Protein Structure Prediction Benchmark",
            },
            "siblings": [{"rfilename": "README.md"}, {"rfilename": "sequences.fasta"}, {"rfilename": "structures/"}],
        },
        "ground_truth": {
            "title": "Protein Structure Prediction Benchmark",
            "description": "Curated benchmark for protein 3D structure prediction from amino acid sequences. Contains 12K protein structures from PDB with sequence-structure pairs, difficulty annotations, and CASP14/15 evaluation splits. Covers single-chain, multi-chain, and disordered protein targets.",
            "subjects": "protein structure, bioinformatics, structure prediction, benchmark, PDB",
            "creator": "bio-bench",
            "publisher": "Computational Biology Benchmarks Consortium",
            "resource_type": "Protein structure prediction benchmark dataset",
            "resource_type_general": "Dataset",
            "publication_year": 2024,
            "size": "12K structures, 8GB",
            "format": "FASTA sequences + PDB structure files",
            "version": "2.1",
            "license": "apache-2.0",
        },
        "domain": "scientific",
    },
]

# Dataset README bodies
DATASET_README_BODIES = {
    "ICICLE-AI/wildlife-camera-traps": """\
# Wildlife Camera Trap Images

Camera trap images of wildlife species from conservation areas across the Midwest United States. The dataset contains approximately 250,000 JPEG images captured by 150 autonomous camera units deployed across Ohio, Indiana, and Illinois state parks and wildlife refuges.

## Dataset Description

Images are annotated with bounding boxes for 45 mammal and bird species. Annotations were created by trained ecologists and verified through a two-stage review process. Species include white-tailed deer, coyote, wild turkey, raccoon, opossum, and red fox among others.

## Usage

Designed for training and evaluating object detection models for ecological monitoring. Supports research in automated wildlife census and biodiversity assessment.
""",
    "ICICLE-AI/midwest-crop-yields": """\
# Midwest Crop Yield Records

County-level crop yield data for corn, soybean, and wheat across 12 Midwest US states from 2000 to 2024. Paired with weather covariates from ERA5 reanalysis including daily temperature, precipitation, soil moisture, and solar radiation.

## Data Description

Each row represents one county-year-crop combination with the reported yield (bushels/acre) and aggregated growing-season weather statistics. Weather features are computed from ERA5 at 0.25-degree resolution and area-averaged to county boundaries.

## Intended Use

Agricultural yield prediction, climate impact assessment, and food security research.
""",
    "ICICLE-AI/era5-midwest-climate": """\
# ERA5 Midwest Climate Reanalysis Subset

Subset of the ECMWF ERA5 reanalysis dataset covering the Midwest United States (36-49°N, 80-104°W) at hourly temporal resolution from 2000 to 2024.

## Variables

Temperature (2m), precipitation, wind speed (10m), relative humidity, downward shortwave radiation, soil temperature (4 levels), soil moisture (4 levels).

## Format

Data is stored in Zarr format with dimensions (time, latitude, longitude). Grid spacing is 0.25 degrees (~28km).
""",
    "ICICLE-AI/policy-documents-qa": """\
# US Policy Documents QA

Question-answer pairs derived from US federal and state policy documents. Covers agriculture policy (Farm Bill sections), environmental regulations (Clean Water Act, Endangered Species Act), and infrastructure planning documents.

## Construction

Questions were generated by domain experts and graduate students. Answers are extractive spans from source documents with citation metadata linking to the original PDF page and paragraph.
""",
}

# ---------------------------------------------------------------------------
# Patra field mappings — values must match schema.json enums
# Source: patra-toolkit-main/patra_toolkit/schema/schema.json
# ---------------------------------------------------------------------------

# category enum from schema.json
PIPELINE_TO_CATEGORY = {
    "image-classification": "classification",
    "object-detection": "computer vision",
    "mask-generation": "computer vision",
    "fill-mask": "natural language processing",
    "text-generation": "natural language processing",
    "text2text-generation": "natural language processing",
    "automatic-speech-recognition": "classification",
    "text-to-audio": "generative modeling",
    "text-to-image": "generative modeling",
    "tabular-classification": "classification",
    "tabular-regression": "regression",
    "zero-shot-image-classification": "classification",
    "image-text-to-text": "natural language processing",
    "image-to-text": "generative modeling",
    "token-classification": "classification",
    "audio-classification": "classification",
    "graph-ml": "graph neural networks",
    "sentence-similarity": "natural language processing",
}

PIPELINE_TO_INPUT_TYPE = {
    "image-classification": "Image",
    "object-detection": "Image",
    "mask-generation": "Image",
    "fill-mask": "Text",
    "text-generation": "Text",
    "text2text-generation": "Text",
    "automatic-speech-recognition": "Audio",
    "text-to-audio": "Text",
    "text-to-image": "Text",
    "tabular-classification": "Tabular",
    "tabular-regression": "Tabular",
    "zero-shot-image-classification": "Multimodal",
    "image-text-to-text": "Multimodal",
    "image-to-text": "Image",
    "token-classification": "Text",
    "audio-classification": "Audio",
    "graph-ml": "Tabular",
    "sentence-similarity": "Text",
}

PIPELINE_TO_OUTPUT = {
    "image-classification": "class label",
    "object-detection": "bounding boxes with class labels",
    "mask-generation": "segmentation mask",
    "fill-mask": "predicted token",
    "text-generation": "generated text",
    "text2text-generation": "generated text",
    "automatic-speech-recognition": "transcribed text",
    "text-to-audio": "generated audio",
    "text-to-image": "generated image",
    "tabular-classification": "class label",
    "tabular-regression": "predicted value",
    "zero-shot-image-classification": "class probabilities",
    "image-text-to-text": "generated text",
    "image-to-text": "generated caption",
    "token-classification": "labeled tokens",
    "audio-classification": "class label",
    "graph-ml": "node/edge predictions",
}

# ai_model.framework enum: sklearn, tensorflow, pytorch, other
LIBRARY_TO_FRAMEWORK = {
    "transformers": "pytorch",
    "ultralytics": "pytorch",
    "diffusers": "pytorch",
    "timm": "pytorch",
    "sentence-transformers": "pytorch",
    "pyannote.audio": "pytorch",
    "scikit-learn": "sklearn",
    "sklearn": "sklearn",
    "xgboost": "other",
    "lightgbm": "other",
    "catboost": "other",
    "tensorflow": "tensorflow",
    "keras": "tensorflow",
    "Mesa": "other",
}

# ai_model.model_type enum: cnn, decision_tree, dnn, rnn, svm, kmeans, llm, random_forest, lstm, gnn, other
MODEL_TYPE_MAP = {
    "resnet": "cnn", "vit": "cnn", "convnext": "cnn", "efficientnet": "cnn",
    "bert": "dnn", "roberta": "dnn", "distilbert": "dnn", "electra": "dnn",
    "llama": "llm", "gpt2": "llm", "mistral": "llm", "falcon": "llm", "qwen": "llm",
    "t5": "dnn", "bart": "dnn",
    "whisper": "dnn",
    "lstm": "lstm", "gru": "rnn",
    "clip": "dnn",
    "gcn": "gnn", "gat": "gnn",
}


def _dataset_to_url(name: str) -> str:
    return f"https://huggingface.co/datasets/{name}"

# ---------------------------------------------------------------------------
# Synthetic README bodies — keyed by repo ID
# Patterns: rich (BibTeX + description), medium (description only),
#           sparse (one line), empty (template placeholder)
# ---------------------------------------------------------------------------

README_BODIES = {
    # --- Rich: full description + BibTeX ---
    "microsoft/resnet-50": """\
# ResNet-50 v1.5

ResNet model pre-trained on ImageNet-1k at resolution 224x224. It was introduced in the paper Deep Residual Learning for Image Recognition by He et al. and achieves 76.1% top-1 accuracy on ImageNet.

## Model description

ResNet (Residual Network) is a convolutional neural network that democratized the concepts of residual learning and skip connections. This enables training much deeper models. This is ResNet v1.5 with 50 layers.

## Intended uses & limitations

You can use the raw model for image classification. See the model hub for fine-tuned versions on a task that interests you.

## Citation

```bibtex
@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={770--778},
  year={2016}
}
```
""",
    "google/vit-base-patch16-224": """\
# ViT Base Patch16 224

Vision Transformer (ViT) model pre-trained on ImageNet-21k and fine-tuned on ImageNet-1k at resolution 224x224. Achieves 84.5% top-1 accuracy using patch size 16.

## Model description

The Vision Transformer uses self-attention over image patches instead of convolutions. Images are split into fixed-size patches, linearly embedded, and processed by a standard Transformer encoder.

## Citation

```bibtex
@inproceedings{dosovitskiy2021image,
  title={An image is worth 16x16 words: Transformers for image recognition at scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and others},
  booktitle={ICLR},
  year={2021}
}
```
""",
    "google-bert/bert-base-uncased": """\
# BERT base model (uncased)

BERT base model (uncased) pre-trained on English text using masked language modeling and next sentence prediction. 12 layers, 768 hidden size, 12 attention heads, 110M parameters.

## Model description

Pretrained model on English language using a masked language modeling (MLM) objective. BERT is a transformers model pretrained on a large corpus of English data in a self-supervised fashion.

## Intended uses & limitations

You can use the raw model for masked language modeling, but it's mostly intended to be fine-tuned on a downstream task like sequence classification, token classification, or question answering.

## Citation

```bibtex
@article{devlin2019bert,
  title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  journal={NAACL},
  year={2019}
}
```
""",
    "meta-llama/Llama-3.1-8B-Instruct": """\
# Llama 3.1 8B Instruct

Llama 3.1 8B Instruct is optimized for dialogue and instruction following with 128K context length. Trained with RLHF on instruction-tuning datasets. Supports 8 languages.

## Model description

Llama 3.1 instruction-tuned models are optimized for dialogue use cases and outperform many open source chat models on common industry benchmarks.

## Citation

```bibtex
@article{dubey2024llama,
  title={The Llama 3 Herd of Models},
  author={Dubey, Abhimanyu and others},
  journal={arXiv preprint arXiv:2407.21783},
  year={2024}
}
```
""",
    "google-t5/t5-small": """\
# T5 Small

T5 Small model with 60M parameters pre-trained on the C4 corpus. Handles translation, summarization, question answering, and classification by casting all tasks as text generation.

## Model description

T5 (Text-To-Text Transfer Transformer) reframes all NLP tasks as a unified text-to-text problem. The model is trained using teacher forcing on a mixture of unsupervised and supervised tasks.

## Citation

```bibtex
@article{raffel2020exploring,
  title={Exploring the limits of transfer learning with a unified text-to-text transformer},
  author={Raffel, Colin and others},
  journal={JMLR},
  year={2020}
}
```
""",
    "openai/whisper-large-v3": """\
# Whisper Large V3

Whisper Large V3 model for multilingual automatic speech recognition with 1.5B parameters. Supports 99 languages. Trained on 5 million hours of weakly supervised audio data.

## Model description

Whisper is a general-purpose speech recognition model trained on a large dataset of diverse audio. It is a multi-task model that can perform multilingual speech recognition, speech translation, and language identification.

## Citation

```bibtex
@article{radford2023robust,
  title={Robust speech recognition via large-scale weak supervision},
  author={Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and others},
  journal={ICML},
  year={2023}
}
```
""",
    "openai/clip-vit-base-patch32": """\
# CLIP ViT-B/32

CLIP model with ViT-B/32 visual encoder for zero-shot image classification. Learns visual concepts from natural language supervision by comparing image and text embeddings.

## Model description

CLIP (Contrastive Language-Image Pre-training) jointly trains an image encoder and a text encoder to predict the correct pairings of a batch of (image, text) training examples. At inference time, learned text and image embeddings enable zero-shot classification.

## Citation

```bibtex
@inproceedings{radford2021learning,
  title={Learning transferable visual models from natural language supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and others},
  booktitle={ICML},
  year={2021}
}
```
""",
    # --- Medium: description but no BibTeX ---
    "ultralytics/yolov8n": """\
# YOLOv8 Nano

YOLOv8 Nano model for real-time object detection trained on the COCO dataset at 640x640 input resolution. Achieves 37.3 mAP on COCO val2017 at 80 FPS on NVIDIA T4.

## Usage

```python
from ultralytics import YOLO
model = YOLO("ultralytics/yolov8n")
results = model("image.jpg")
```

## Training details

Trained on COCO train2017 with standard augmentation pipeline. See args.yaml for full hyperparameters.
""",
    "suno/bark": """\
# Bark

Bark is a transformer-based text-to-audio model that generates realistic speech, music, and sound effects. Supports multiple languages and speakers with voice cloning capability.

## Usage

```python
from transformers import AutoProcessor, BarkModel
processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark")
```
""",
    "stabilityai/stable-diffusion-xl-base-1.0": """\
# Stable Diffusion XL Base 1.0

SDXL Base 1.0 generates 1024x1024 images from text prompts. 3.5B parameter UNet with dual text encoders (CLIP ViT-L and OpenCLIP ViT-bigG).

## Model description

SDXL is a latent diffusion model with significantly improved image quality and prompt adherence compared to previous Stable Diffusion versions. Uses a larger UNet backbone and a second text encoder for improved text understanding.
""",
    # --- Sparse ICICLE: one-liner or minimal ---
    "ICICLE-AI/wildlife-detector-v2": """\
# Wildlife Detector V2

YOLO-based wildlife detector trained on camera trap imagery for ecological monitoring. Detects and classifies common North American wildlife species.
""",
    "ICICLE-AI/policy-ner-agriculture": """\
# Policy NER Agriculture

Named entity recognition model for extracting policy-relevant terms from agricultural documents.
""",
    "ICICLE-AI/birdsong-classifier-v1": """\
# Birdsong Classifier V1

Audio classifier for identifying bird species from field recordings. Designed for ecological survey automation.
""",
    "ICICLE-AI/satellite-crop-captioner": """\
# Satellite Crop Captioner

Image captioning model for describing crop conditions in satellite imagery.
""",
    "ICICLE-AI/supply-chain-gnn": """\
# Supply Chain GNN

GNN model for predicting supply chain disruption propagation in food distribution networks.
""",
    # --- Tabular: minimal ---
    "scikit-learn-mojos/randomforest-iris": """\
# Random Forest Iris

Random Forest classifier for Iris species classification. 100 estimators, max_depth=5. Achieves 97% test accuracy.
""",
    "xgboost-community/xgb-housing-regressor": """\
# XGBoost Housing Regressor

XGBoost gradient boosting regressor for California housing price prediction. 200 estimators, learning rate 0.1. RMSE 0.46 on test set.
""",
    # --- Empty / template ---
    "ICICLE-AI/crop-yield-predictor": "",
    "ICICLE-AI/water-distribution-abm": "",
    "ICICLE-AI/climate-downscaling-unet": "",
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    dataset = []
    input_cards_dir = Path(__file__).parent / "data" / "inputs" / "input_cards"
    input_cards_dir.mkdir(exist_ok=True)
    input_ds_dir = Path(__file__).parent / "data" / "inputs" / "input_datasheets"
    input_ds_dir.mkdir(exist_ok=True)
    md_dir = Path(__file__).parent / "data" / "inputs" / "input_markdowns"
    md_dir.mkdir(exist_ok=True)

    # Model cards (IDs 04-23)
    for i, entry in enumerate(CARDS):
        card_id = i + 4
        hf_card = entry["sparse"]
        repo_id = hf_card["id"]
        readme = README_BODIES.get(repo_id, "")

        dataset.append({
            "id": card_id,
            "asset_type": "model_card",
            "repo_id": repo_id,
            "domain": entry["domain"],
            "hf_api_response": hf_card,
            "readme_body": readme,
            "patra_ground_truth": entry["ground_truth"],
        })

        slug = repo_id.replace("/", "_").lower()
        with open(input_cards_dir / f"{card_id:02d}_{slug}.json", "w") as f:
            json.dump(hf_card, f, indent=2)
        with open(md_dir / f"{card_id:02d}_{slug}.md", "w") as f:
            f.write(readme)

    # Datasheets (IDs 24-33)
    for i, entry in enumerate(DATASHEETS):
        ds_id = i + 24
        hf_ds = entry["sparse"]
        repo_id = hf_ds["id"]
        readme = DATASET_README_BODIES.get(repo_id, "")

        dataset.append({
            "id": ds_id,
            "asset_type": "datasheet",
            "repo_id": repo_id,
            "domain": entry["domain"],
            "hf_api_response": hf_ds,
            "readme_body": readme,
            "patra_ground_truth": entry["ground_truth"],
        })

        slug = repo_id.replace("/", "_").lower()
        with open(input_ds_dir / f"{ds_id:02d}_{slug}.json", "w") as f:
            json.dump(hf_ds, f, indent=2)
        with open(md_dir / f"{ds_id:02d}_{slug}.md", "w") as f:
            f.write(readme)

    out_path = Path(__file__).parent / "data" / "inputs" / "synthetic_hf_cards.json"
    with open(out_path, "w") as f:
        json.dump(dataset, f, indent=2)

    mc_count = sum(1 for e in dataset if e["asset_type"] == "model_card")
    ds_count = sum(1 for e in dataset if e["asset_type"] == "datasheet")
    print(f"Generated {len(dataset)} synthetic entries ({mc_count} model cards, {ds_count} datasheets)")
    print(f"  Dataset:       {out_path}")
    print(f"  Model cards:   {input_cards_dir}/")
    print(f"  Datasheets:    {input_ds_dir}/")
    print()
    for entry in dataset:
        hf = entry["hf_api_response"]
        tag = entry["asset_type"][:2].upper()
        has_pipeline = "Y" if hf.get("pipeline_tag") else "N"
        has_readme = "Y" if entry["readme_body"] else "N"
        print(f"  [{entry['id']:2d}] [{tag}] {entry['repo_id']:<45s}  readme={has_readme}  ({entry['domain']})")


if __name__ == "__main__":
    main()
