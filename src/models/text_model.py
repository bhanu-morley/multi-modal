"""
DeBERTa-based Text Emotion Detection Model
-------------------------------------------
A fine-tuned DeBERTa-v3-base model for emotion classification.
Outputs emotion label + confidence score.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from ..config.settings import TEXT_MODEL_CONFIG

# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_DIR = TEXT_MODEL_CONFIG["model_path"]

# Select GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# 2. MODEL LOADING
# ============================================================

# Load config to ensure correct label order
config = AutoConfig.from_pretrained(MODEL_DIR)
id2label = config.id2label
EMOTION_CLASSES = [id2label[i] for i in range(len(id2label))]

try:
    # Use slow tokenizer to avoid PyPreTokenizerTypeWrapper bug
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        use_fast=False,              # <<--- CRITICAL FIX
        local_files_only=True        # ensure we use your saved model
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR,
        local_files_only=True
    )

    model.to(device)
    model.eval()
    print("DeBERTa model loaded successfully.")
except Exception as e:
    print(f"Failed to load DeBERTa model: {e}")
    print("Fix steps:")
    print("Delete tokenizer.json in model folder")
    print("Run: pip install \"tokenizers<0.14\"")
    tokenizer, model = None, None

# ============================================================
# 3. EMOTION PREDICTION
# ============================================================

def predict_text_emotion(text: str) -> dict:
    """Predict emotion and confidence from text input."""
    if tokenizer is None or model is None:
        return {"error": "Model not loaded"}

    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=TEXT_MODEL_CONFIG["maxlen"]
    ).to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()[0]

    # Predictions
    pred_idx = int(np.argmax(probs))
    emotion = EMOTION_CLASSES[pred_idx]
    confidence = float(np.max(probs))

    print("\nEmotion Probabilities:")
    for cls, p in zip(EMOTION_CLASSES, probs):
        print(f"  {cls:<10}: {p:.3f}")

    return {
        "emotion": emotion,
        "confidence": round(confidence, 3)
    }


# ============================================================
# 4. STANDALONE TEST
# ============================================================

# if __name__ == "__main__":
#     sample_texts = [
#         "I am so happy today!",
#         "I feel really sad and lonely.",
#         "That movie terrified me.",
#         "This is absolutely disgusting!",
#         "I'm not angry at all.",
#         "Wow, I didn't expect that!"
#     ]

#     for txt in sample_texts:
#         result = predict_text_emotion(txt)
#         print(f"\nInput: {txt}")
#         print(f"Predicted Emotion: {result['emotion']} | Confidence: {result['confidence']:.2f}")
#         print("-" * 60)
