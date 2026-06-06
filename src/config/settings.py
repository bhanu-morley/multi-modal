
# This file centralizes all configuration settings used across
# the Multimodal Sentiment Analysis project (Text + Vision).
# It defines global constants, paths, model configurations,
# supported formats, and UI theme preferences.


import os
from pathlib import Path

# ============================================================
# 1. BASE DIRECTORY SETUP
# ------------------------------------------------------------
#
# Defines BASE_DIR as the root of the project directory.
# Useful for dynamically constructing paths to model files,
# datasets, and assets without hardcoding absolute paths.
#
# ============================================================

BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# ============================================================
# 2. VISION MODEL CONFIGURATION (VGG16)
# ------------------------------------------------------------
#
# Settings for the Vision Model (CNN-based using VGG16).
# Includes model paths, label mapping, and checkpoint details.
#
# ============================================================

# 2.1 Face model configuration (VGG16 trained on fer2013 & ck+)
FACE_MODEL_CONFIG = {
    "model_name": "VGG16",      # CNN model name
    "num_classes": 7,           # Number of emotion classes (e.g., FER2013, CK+)
    "checkpoint_path": os.path.join(BASE_DIR, "src/trained_models", "vgg16_fer_ck_adv.pth"),  # Pretrained weights
    "class_labels": ["angry","disgust","fear","happy","sad","surprise","neutral"]

}


# ============================================================
# 3. TEXT MODEL CONFIGURATION (DeBERTa)
# ------------------------------------------------------------
#
# Settings for the DeBERTa-based text emotion classification model.
# Paths are dynamically constructed relative to BASE_DIR.
#
# ============================================================

# 3.1 Text model configuration (DeBERTa trained on GoEmotions)
TEXT_MODEL_CONFIG = {
    "model_name": "DeBERTa Emotion Model",    # Descriptive name for text model
    "model_path": os.path.join(BASE_DIR, "src", "trained_models", "deberta_emotion_model_final"),     # DeBERTa model file
    "num_classes": 7,
    "maxlen": 128      # Maximum input sequence length used during training
}


# ============================================================
# 4. IMAGE PREPROCESSING PARAMETERS
# ------------------------------------------------------------
#
# Standard normalization parameters for image inputs.
# Matches ImageNet preprocessing conventions.
#
# ============================================================

IMAGE_TRANSFORMS = {
    "resize": 224,  # Image resized to 224x224 pixels
    "center_crop": 224,  # Ensures central crop if necessary
    "normalize_mean": [0.485, 0.456, 0.406],  # Standard RGB mean values
    "normalize_std": [0.229, 0.224, 0.225],   # Standard RGB standard deviations
}


# ============================================================
# 5. SENTIMENT LABEL MAPPINGS
# ------------------------------------------------------------
#
# Maps emotion indices from model outputs to human-readable labels.
# Supports 4-, and 7-class emotion models.
#
# ============================================================

SENTIMENT_MAPPINGS = {
    4: {0: "Angry", 1: "Sad", 2: "Happy", 3: "Neutral"},  # 4-class model
    7: {  # 7-class model (FER2013 / CK+)
        0: "Angry",
        1: "Disgust",
        2: "Fear",
        3: "Happy",
        4: "Sad",
        5: "Surprise",
        6: "Neutral",
    },
}


# ============================================================
# 6. DIRECTORY PATH SHORTCUTS
# ------------------------------------------------------------
#
# Common directories used throughout the project for easy imports.
# ============================================================

MODELS_DIR = BASE_DIR / "models"    # Model source code directory
SRC_DIR = BASE_DIR / "src"          # Main source directory
UI_DIR = SRC_DIR / "ui"             # UI-specific components folder
