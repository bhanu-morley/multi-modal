# Multimodal Sentiment Analysis - Gradio Version
# ============================================================================================================---
# This script wires together:
#  - Gradio UI (three tabs: text, face, fusion)
#  - Model inference wrappers (text, face, fusion)
#  - Image storage (GridFS)
#  - Audit logging (MongoDB records collection)
#
# Keep model functions (predict_text_emotion, predict_vision_sentiment, predict_fused_sentiment)
# in src.models.* — they must return the documented dict/list shapes used below.
# ============================================================================================================---

import gradio as gr
from PIL import Image
from datetime import datetime
import io
import os
from pymongo import MongoClient
from gridfs import GridFS
from dotenv import load_dotenv

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"      # Disable oneDNN numerical diff warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"       # 0=all logs, 3=errors only

# ======================================================
# 0. Environment + DB Init
# ======================================================
# Load environment variables from a .env file (MONGO_URI, etc.).
# load_dotenv() will read the .env file and set values in os.environ.
load_dotenv()


# Create a MongoDB client from the MONGO_URI environment variable.
# If MONGO_URI is missing you will get a runtime error from MongoClient or connection attempts.
client = MongoClient(os.getenv("MONGO_URI"))

# Use / create the database named "multi-modal" (matches your other modules).
db = client["multi-modal"]

# GridFS provides chunked file storage inside MongoDB for binary files (images).
fs = GridFS(db)


# ======================================================
# 1. Import Model Wrappers
# ======================================================
# 
# These modules should expose the following interfaces:
# - predict_text_emotion(text:str) -> dict like {"emotion": "happy", "confidence": 0.92}
# - predict_vision_sentiment(image:PIL.Image) -> list[dict] like [{"face_id":0, "emotion":"happy", "confidence":0.9}, ...]
# - predict_fused_sentiment(text:str, image:PIL.Image) -> tuple (emotion:str, confidence:float)
#
# Keeping model logic separate from UI lets you swap models without changing the UI code.

from src.models.fusion_model import predict_fused_sentiment
from src.models.text_model import predict_text_emotion
from src.models.face_model import predict_vision_sentiment

# ======================================================
# 2. GridFS Image Storage
# ======================================================

def store_image_gridfs(img: Image.Image):
    """
    Save a PIL.Image into GridFS and return the generated file id.
    - Converts image bytes to JPEG in-memory to keep storage consistent.
    - Returns a GridFS file id (ObjectId) that you can store in records.
    """
    # Use BytesIO to avoid writing to disk—fast and safe inside multi-user apps.
    buf = io.BytesIO()
    # Save as JPEG with high quality to preserve visual details for later inspection.
    img.save(buf, format="JPEG", quality=95)
    # fs.put expects raw bytes; we attach a helpful filename with a timestamp.
    return fs.put(buf.getvalue(), filename=f"img_{datetime.now().isoformat()}.jpg")

# ======================================================
# 3. Database Logging
# ======================================================

def save_record(record: dict):
    """
    Append a record document to the 'records' collection.
    Keep documents small and typed:
      - mode: "text" | "face" | "fusion"
      - timestamp: datetime
      - any prediction fields (strings / floats)
      - optional: image_file_id (GridFS id)
      - optional: source (e.g., "gradio_app")
    Note: This is synchronous and will block the request while writing.
    """
    db.records.insert_one(record)


# ======================================================
# 4. Callback: Text Sentiment
# ======================================================

def text_sentiment(text: str):
    """
    Called by the Text Sentiment tab.
    - Validates input
    - Calls text model wrapper
    - Saves a minimal audit record to MongoDB
    - Returns display-friendly strings for Gradio outputs
    """
    # Basic validation to avoid calling model on empty input
    if not text or not text.strip():
        return "Please enter text.", ""

    # Call the text model — expected dict: {"emotion": str, "confidence": float}
    result = predict_text_emotion(text)

    # If the model wrapper returns an error-like payload, pass it back to the UI
    if isinstance(result, dict) and "error" in result:
        return result["error"], ""

    # Persist a structured record for later analysis / dashboard
    save_record({
        "mode": "text",
        "text_input": text,
        "text_prediction": result["emotion"],
        "text_confidence": float(result["confidence"]),
        "timestamp": datetime.now(),
        "source": "gradio_app"
    })

    # Return two simple text outputs: emotion and confidence (formatted)
    return f"Emotion: {result['emotion']}", f"Confidence: {result['confidence']:.2f}"


# ======================================================
# 5. Callback: Face Sentiment
# ======================================================

def face_sentiment(image: Image.Image):
    """
    Called by the Face Emotion tab.
    - Accepts a PIL image (Gradio returns PIL when type='pil')
    - Calls vision model wrapper which may return:
        * a list of face results: [{"face_id": int, "emotion": str, "confidence": float}, ...]
        * or a single-element list/dict indicating no face (face_id == 0) depending on your model
    - Stores the image in GridFS and logs the best face prediction
    - Returns a multi-line text summary for the UI
    """

    if image is None:
        return "Please upload an image."

    # Model returns list of face detection results
    results = predict_vision_sentiment(image)

    # If model indicates "no face" with a special code, handle gracefully
    if isinstance(results, list) and len(results) == 1 and results[0].get("face_id") == 0:
        return "No face detected in image."

    # Choose the face with highest confidence as the representative result
    best = max(results, key=lambda x: x.get("confidence", 0.0))

    # Persist the original image and get a GridFS id back
    img_id = store_image_gridfs(image)

    # Save structured record referencing the stored image
    save_record({
        "mode": "face",
        "face_prediction": best["emotion"],
        "face_confidence": float(best["confidence"]),
        "image_file_id": img_id,
        "timestamp": datetime.now(),
        "source": "gradio_app"
    })

    # Build a readable multiline summary for all detected faces (if multiple)
    txt = "\n".join([
        f"Face {r.get('face_id', '?')} → {r.get('emotion', 'Unknown')} (Conf: {r.get('confidence', 0.0):.2f})"
        for r in results
    ])
    return txt


# ======================================================
# 6. Callback: Fusion Sentiment
# ======================================================

def fused_sentiment(text: str, img: Image.Image):
    """
    Called by the Fusion tab.
    - Validates both inputs are present
    - Converts image to RGB to make model input deterministic
    - Calls fusion model to compute a combined emotion + confidence
    - Also runs the single-modality models to log their outputs for audit/analysis
    - Stores image and logs a fused record
    - Returns a single-line result string for the UI
    """

    if not text or img is None:
        return "Please provide text and image input."

    # Ensure image has a consistent color format
    image = img.convert("RGB")

    # Predict fused emotion via your fusion model.
    # Expected: (emotion_str, confidence_float)
    fused_emotion, fused_conf = predict_fused_sentiment(text, image)

    # Also collect single-modality predictions for transparency / debugging
    text_res = predict_text_emotion(text)
    face_res = predict_vision_sentiment(image)
    # face_res is expected as a list; pick the highest-confidence face
    best = max(face_res, key=lambda x: x.get("confidence", 0.0))

    # Store the image and persist the full fusion record
    img_id = store_image_gridfs(image)

    save_record({
        "mode": "fusion",
        "text_input": text,
        "text_prediction": text_res["emotion"],
        "text_confidence": float(text_res["confidence"]),
        "face_prediction": best["emotion"],
        "face_confidence": float(best["confidence"]),
        "fused_emotion": fused_emotion,
        "fused_confidence": float(fused_conf),
        "image_file_id": img_id,
        "timestamp": datetime.now(),
        "source": "gradio_app"
    })

    return f"Fused Sentiment: {fused_emotion} (Conf: {fused_conf:.2f})"


# ======================================================
# 7. UI / Layout
# ======================================================
# CSS to constrain layout and control how images render in the Gradio UI.

css = """
#app-container {
    max-width: 900px;
    margin: auto;
    padding: 30px;
}
.gradio-container img {
    object-fit: contain !important;
    max-width: 350px !important;
    max-height: 250px !important;
}
button[data-testid*="all-btn"] {
    background-color: #007BFF !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 18px !important;
    font-weight: 600 !important;
    transition: background-color 0.25s ease;
}
button[data-testid*="button"]:hover {
    background-color: #00AEEF !important;
}
"""

# Build the Gradio Blocks UI. Using Blocks gives explicit layout control, and Tabs
# keep the three workflows (Text, Face, Fusion) separate and easy to maintain.
with gr.Blocks(title="Multimodal Emotion Analysis", css=css) as app:
    with gr.Column(elem_id='app-container'):

        gr.Markdown("### Multimodal Sentiment Analysis")
        gr.Markdown("Fusion of Text + Facial Emotion Recognition")

        with gr.Tabs():

            # Text Sentiment Tab
            with gr.Tab("Text Sentiment"):
                # Simple textbox for user text input and a button to trigger analysis
                text = gr.Textbox(label="Enter Text")
                btn = gr.Button("Analyze Text", elem_classes="[all-btn]")
                out, conf = gr.Textbox(label="Classified emotion"), gr.Textbox(label="Confidence score")
                # Wire up the button to the text_sentiment callback
                btn.click(text_sentiment, text, [out, conf])

            # Face Emotion Tab
            with gr.Tab("Face Emotion"):
                # Gradio provides PIL unless type is changed; we use 'pil' to keep model code simple
                img = gr.Image(type="pil", sources=["upload", "clipboard"], label="Upload Image")
                btn2 = gr.Button("Analyze Face", elem_classes="[all-btn]")
                out2 = gr.TextArea(label="Detected emotion")
                btn2.click(face_sentiment, img, out2)

            # Fusion Tab
            with gr.Tab("Fusion"):
                fimg = gr.Image(type="pil", sources=["upload", "clipboard"], label="Upload Image")
                ftext = gr.Textbox(label="Enter Text")
                btn3 = gr.Button("Run Fusion", elem_classes="[all-btn]")
                fout = gr.Textbox(label="Fused emotion result")
                btn3.click(fused_sentiment, [ftext, fimg], fout)

            # Info Tab
            with gr.Tab("Info"):
                gr.Markdown("""
                ### Fusion Strategy Info
                - Text Transformer Model
                - CNN Facial Emotion Model
                - Weighted Confidence Fusion
                """)

os.environ["GRADIO_DISABLE_API_DOCS"] = "1"

app.queue().launch(server_port=7860, share=False)
