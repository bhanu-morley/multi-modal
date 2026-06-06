"""
Fusion Model — Combines predictions from:
  1. Text emotion model (DeBERTa)
  2. Vision emotion model (CNN / FER)

Supports:
  - Multiple face detection
  - Half-face / partial face handling
  - Weighted confidence fusion
"""

import logging
from typing import Tuple, Optional
from PIL import Image

# Import individual models
from .text_model import predict_text_emotion
from .face_model import predict_vision_sentiment

# ============================================================
# LOGGER SETUP
# ============================================================
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ============================================================
# FUSION FUNCTION
# ============================================================

def predict_fused_sentiment(
    text: Optional[str] = None,
    image: Optional[Image.Image] = None,
) -> Tuple[str, float]:
    """
    Fuse text and image emotion predictions using weighted averaging.
    Handles multi-face, half-face, and missing modalities gracefully.
    """

    results = []  # [(modality, emotion, confidence)]
    modality_weights = {"Text": 0.3, "Vision": 0.7}

    # ----------------------------
    # TEXT PREDICTION
    # ----------------------------
    if text:
        try:
            text_result = predict_text_emotion(text)
            if isinstance(text_result, dict):
                text_emotion = text_result.get("emotion", "Unknown")
                text_conf = float(text_result.get("confidence", 0.0))
            else:
                text_emotion, text_conf = text_result  # if older tuple-return version

            results.append(("Text", text_emotion, text_conf))

        except Exception as e:
            logger.error(f"Text model error: {e}")
            results.append(("Text", "Error", 0.0))

    # ----------------------------
    # VISION PREDICTION
    # ----------------------------
    vision_emotion, vision_conf = "Unknown", 0.0  # ensure always defined

    if image:
        try:
            vision_output = predict_vision_sentiment(image)

            # Case 1: Multiple detected faces (list)
            if isinstance(vision_output, list):
                if not vision_output:
                    logger.warning("Vision model returned an empty face list.")
                else:
                    # pick the most confident face
                    best_face = max(vision_output, key=lambda x: x.get("confidence", 0.0))
                    vision_emotion = best_face.get("emotion", "Unknown")
                    vision_conf = float(best_face.get("confidence", 0.0))
                    logger.info(f"Most confident face: {vision_emotion} ({vision_conf:.2f})")

            # Case 2: Single (emotion, confidence) tuple
            elif isinstance(vision_output, tuple) and len(vision_output) == 2:
                vision_emotion, vision_conf = vision_output

            # Case 3: Unexpected format
            else:
                logger.warning(f"Unexpected vision output type: {type(vision_output)}")

        except Exception as e:
            logger.error(f"Vision model failed: {e}")
            vision_emotion, vision_conf = "Error", 0.0

        results.append(("Vision", vision_emotion, vision_conf))

    # ----------------------------
    # NO INPUTS CASE
    # ----------------------------
    if not results:
        logger.warning("No input provided.")
        return "No inputs provided", 0.0

    # ----------------------------
    # FUSION LOGIC
    # ----------------------------
    # emotion_scores = {}
    # weighted_conf_sum = 0.0
    # weight_sum = 0.0

    text_data = next(((m, e, c) for m, e, c in results if m == "Text"), None)
    vision_data = next(((m, e, c) for m, e, c in results if m == "Vision"), None)

    # Default
    final_emotion = None
    final_conf = 0.0

    if text_data and vision_data:
        _, text_e, text_c = text_data
        _, vis_e, vis_c = vision_data

        # Normalize confidences
        text_c = float(text_c)
        vis_c = float(vis_c)

        # Adaptive weight scaling based on reliability
        # Higher confidence → higher weight
        text_weight = 0.3 * (0.5 + text_c)   # ranges 0.15 - 0.45
        vis_weight = 0.7 * (0.5 + vis_c)    # ranges 0.35 - 0.98

        # Normalize weights to sum to 1
        total_w = text_weight + vis_weight
        text_w = text_weight / total_w
        vis_w = vis_weight / total_w

        # Weighted probability
        base_conf = (text_w * text_c) + (vis_w * vis_c)

        if text_e == vis_e:
            # Synergy boost
            synergy = (text_c * vis_c) ** 0.5 * 0.7
            final_conf = min(1.0, base_conf + synergy)
            final_emotion = text_e
        else:
            # Disagreement penalty
            penalty = abs(text_c - vis_c) * 0.4
            final_conf = max(0.0, base_conf - penalty)
            final_emotion = text_e if text_c > vis_c else vis_e
    else:
        # Only text or only face — fallback
        (m, e, c) = text_data or vision_data
        final_emotion = e
        final_conf = float(c)

    final_conf = round(final_conf, 3)

    logger.info(f"Fused Emotion: {final_emotion} ({final_conf})")
    return final_emotion, final_conf

# ============================================================
# STRATEGY INFO
# ============================================================

def get_fusion_strategy_info() -> dict:
    """Provides metadata about fusion strategy."""
    return {
        "strategy_name": "Weighted Ensemble Fusion",
        "description": "Combines text (DeBERTa) and vision (CNN) emotion predictions with multi-face support.",
        "modality_weights": {"Text": 0.1, "Vision": 0.9},
        "fusion_method": "Weighted confidence-based averaging",
        "advantages": [
            "Handles multiple and half-face detections",
            "Robust against missing or failed modalities",
            "Balances facial and textual cues effectively",
        ],
    }
