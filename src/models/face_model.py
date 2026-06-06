# Handles emotion detection from facial images using:
# - FER (EfficientNet-based emotion recognition)
# - CNN: Custom VGG16 vision model trained on fer and ck+ datasets (for fallback)


import os
import cv2
import torch
import torch.nn as nn
import logging
import numpy as np
from PIL import Image
from torchvision import models
from fer.fer import FER
from torchvision import transforms

from ..config.settings import FACE_MODEL_CONFIG

# ====================================================================
# 1. Logger Configuration
# ====================================================================

# A logger helps track what's happening inside this module.
# All key events such as loading models, predictions, errors will be logged.

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Attach a handler only if it doesn't already exit (prevent duplicates)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)


# =======================================================================
# 2. FUNCTION: Load Pretrained VGG16 Vision Model (Optional for fallback)
# -----------------------------------------------------------------------
#
# Purpose: 
#   Loads a pretrained VGG16 model locally that trained on-
#   fer2013 and ck+ datasets from a checkpoint path.
#
# Returns:
#   model (torch.nn Module) - Loads PyTorch (.pth) model
#   device (torch.device) - CPU or GPU used for inference
#
# =======================================================================

def load_vision_model():
    """
    Load a pretrained VGG16 model for emotion classification.
    Falls back gracefully if model or checkpoint not found.
    """
    try:
        # Select device based on hardware availability (CPU or GPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load configuartion parameters (checkpoint path and class count)
        checkpoint_path = FACE_MODEL_CONFIG["checkpoint_path"]
        num_classes = FACE_MODEL_CONFIG["num_classes"]

        # Check if checkpoint exists before loading
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Initialize base VGG16 architecture (no pretrained weights)
        model = models.vgg16(weights=None)

        # Modify the last fully-connected layer to match the number of emotion classes
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

        # Load trained model weights from the checkpoint
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)

        # Move model to GPU/CPU and set to evalution mode
        model.to(device)
        model.eval()
        print("Face model loaded successfully.")

        # Log successful handling
        logger.info(f"VGG16 vision model loaded successfully on {device}")
        return model, device

    except Exception as e:
        # Log errors and return None if model fails to load
        logger.error(f"Failed to load VGG16 model: {e}")
        return None, None
    

# ====================================================================
# 3. HELPER FUNCTION: Predict Emotion with VGG16 (Fallback)
# ====================================================================

def predict_with_vgg(model, device, face_img):
    """
    Run emotion prediction using the fallback VGG16 model.
    Input: face_img (BGR numpy image)
    Returns: (emotion, confidence)
    """
    try:
        if model is None:
            return "Unknown", 0.0

        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        tensor = preprocess(face_img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            conf, pred_class = torch.max(probs, 0)

        idx_to_label = FACE_MODEL_CONFIG.get("class_labels", None)
        if idx_to_label:
            emotion = idx_to_label[pred_class.item()]
        else:
            emotion = f"Class_{pred_class.item()}"

        return emotion.capitalize(), float(conf.item())

    except Exception as e:
        logger.error(f"VGG16 fallback prediction failed: {e}")
        return "Unknown", 0.0



# ====================================================================
# 3. Emotion Prediction Using FER (EfficientNet)
# --------------------------------------------------------------------
#
# Purpose:
#   Detect emotion from an input image using the FER library.
# 
# Steps:
#   1. Load image in correct format
#   2. Detect face(s) using MTCNN (via FER)
#   3. Predict top emotion and confidence
#   4. Handle missing faaces gracefully
#
# Returns:
#   (emotion_name: str, confidence: float)
#   Example → ("Happy", 0.89)
#
# ====================================================================

def predict_vision_sentiment(image_source):
    """
    Predict emotion from an image using FER (EfficientNet + MTCNN).
    Handles image input from path, Streamlit upload, or PIL Image.
    """
    try:
        # 3.1 Initialize FER emotion detector 
        # (uses MTCNN for face detection & EfficientNet for emotion classification)
        detector = FER(mtcnn=True)

        # 3.2 Handle multiple input formats
        if isinstance(image_source, str):  
            # Case 1: A file path string → read image via OpenCV
            img = cv2.imread(image_source)

        elif hasattr(image_source, "read"):  
            # Case 2: Streamlit UploadedFile → convert bytes to OpenCV image
            file_bytes = np.asarray(bytearray(image_source.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        elif isinstance(image_source, Image.Image):  
            # Case 3: PIL Image → convert to OpenCV BGR format
            img = cv2.cvtColor(np.array(image_source), cv2.COLOR_RGB2BGR)

        else:
            # Unsupported input format
            raise ValueError(f"Unsupported image type: {type(image_source)}")


        # 3.3 Check if image was loaded correctly
        if img is None:
            logger.warning("Could not read the image or invalid path provided.")
            return [{"face_id": 0, "emotion": "Unknown", "confidence": 0.0}]
        
        results = detector.detect_emotions(img)
        if not results:
            logger.warning("No faces detected!")
            return [{"face_id": 0, "emotion": "Unknown", "confidence": 0.0}]

        # Sort faces left → right
        results.sort(key=lambda x: x["box"][0])

        # Load fallback VGG model (once)
        vgg_model, device = load_vision_model()

        final_predictions = []
        for idx, face in enumerate(results, 1):
            (x, y, w, h) = face["box"]
            emotions = face["emotions"]

            # Primary FER prediction
            top_emotion = max(emotions, key=emotions.get)
            confidence = float(emotions[top_emotion])
            model_used = "FER+MTCNN"

        # If confidence is low, retry with VGG16
            if confidence < 0.3:
                logger.info(f"Low confidence ({confidence:.2f}) for Face {idx} — retrying with VGG16 fallback.")
                face_crop = img[y:y+h, x:x+w]
                fallback_emotion, fallback_conf = predict_with_vgg(vgg_model, device, face_crop)
                if fallback_conf > confidence:
                    top_emotion, confidence = fallback_emotion, fallback_conf
                    model_used ="VGG16 Fallback"

            final_predictions.append({
                "face_id": idx,
                "emotion": top_emotion.capitalize(),
                "confidence": round(confidence, 2),
                "box": (x, y, w, h),
                "model_used": model_used
            })

            logger.info(f"Face {idx}: {top_emotion.capitalize()} ({confidence:.2f}) | Model Used: {model_used}")

        return final_predictions

    except Exception as e:
        logger.error(f"Vision sentiment prediction failed: {e}")
        return [{"face_id": 0, "emotion": "Unknown", "confidence": 0.0}]



