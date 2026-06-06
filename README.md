# Multimodal Emotion Recognition System using CNN and DeBERTa

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12.9-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)
![MongoDB](https://img.shields.io/badge/MongoDB-Database-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-teal.svg)
![License](https://img.shields.io/badge/License-MIT-blue)

### A Deep Learning-Based Multimodal Emotion Recognition Framework

**Combining Facial Expression Analysis and Text Emotion Understanding for Robust Emotion Prediction**

</div>

---

# Overview

The **Multimodal Emotion Recognition System** is an intelligent AI application that combines **Computer Vision** and **Natural Language Processing (NLP)** to accurately identify human emotions from both **facial expressions** and **textual input**.

Traditional emotion recognition systems rely on a single modality, such as facial images or text, which often leads to inaccurate predictions due to missing contextual information. Our proposed multimodal framework overcomes this limitation by integrating:

* **CNN-based Facial Emotion Recognition**
* **DeBERTa-based Text Emotion Classification**
* **Confidence-Aware Fusion Engine**
* **MongoDB-based Data Storage**
* **Admin Dashboard for Monitoring and Analytics**

The system generates a final emotion prediction by intelligently combining outputs from both modalities, resulting in improved accuracy, robustness, and contextual understanding.

---

# Project Objectives

* Develop a multimodal emotion recognition framework.
* Perform facial emotion detection using CNN-based models.
* Perform text emotion classification using DeBERTa Transformer.
* Implement adaptive confidence-based fusion.
* Store predictions and user data in MongoDB.
* Provide a scalable architecture for future deployment.
* Improve prediction reliability compared to single-modality systems.

---

# Key Features

### Text Emotion Recognition

* Fine-tuned DeBERTa Transformer
* Context-aware emotion prediction
* Handles complex sentence structures
* Multi-class emotion classification

### Facial Emotion Recognition

* FER-based facial emotion detection
* MTCNN face localization
* CNN/VGG16-based fallback prediction
* Supports multiple facial expressions

### Multimodal Fusion

* Confidence-aware weighted fusion
* Handles disagreement between modalities
* Improves final prediction reliability
* Dynamic confidence calculation

### Database Integration

* MongoDB storage
* GridFS image management
* Prediction history tracking
* User activity logging

### Admin Dashboard

* Secure administration panel
* Record management
* Prediction monitoring
* Image visualization

---

# System Architecture

```text
                ┌───────────────────┐
                │     User Input    │
                │ Text + Face Image │
                └─────────┬─────────┘
                          │
          ┌───────────────┴───────────────┐
          │                               │
          ▼                               ▼

 ┌─────────────────┐          ┌─────────────────┐
 │ Text Processing │          │ Face Processing │
 │    DeBERTa      │          │ CNN / FER Model │
 └────────┬────────┘          └────────┬────────┘
          │                            │
          ▼                            ▼

     Text Emotion                Face Emotion
     Confidence                  Confidence

          └───────────┬────────────┘
                      ▼

          ┌───────────────────────┐
          │ Adaptive Fusion Layer │
          └───────────┬───────────┘
                      ▼

          Final Emotion Prediction
                      │
                      ▼

              MongoDB Storage
                      │
                      ▼

              Admin Dashboard
```

---

# 📂 Project Structure

```text
multi-modal/
│
├── datasets/
│   ├── ckplus/
│   ├── fer2013/
│   └── GoEmotions/
│
├── src/
│   ├── config/
│   │   └── settings.py
│   │
│   ├── database/
│   │   └── db.py
│   │
│   ├── models/
│   │   ├── face_model.py
│   │   ├── text_model.py
│   │   └── fusion_model.py
│   │
│   └── trained_models/
│       ├── deberta_emotion_model_final/
│       └── vgg16_fer_ck_adv.pth
│
├── app.py
├── admin.py
├── requirements.txt
├── .env
└── README.md
```

---

# Technology Stack

## Programming Language

* Python

## Deep Learning Frameworks

* PyTorch
* HuggingFace Transformers

## NLP Model

* DeBERTa (Decoding-enhanced BERT)

## Computer Vision

* CNN
* VGG16
* FER
* MTCNN
* OpenCV

## Backend Framework

* FastAPI

## Database

* MongoDB
* GridFS

## Data Processing

* NumPy
* Pandas
* NLTK

## Model Training

* Torch
* Transformers
* Scikit-learn

---

# Datasets Used

## 1. FER2013

Used for facial emotion recognition.

Emotions:

* Angry
* Disgust
* Fear
* Happy
* Sad
* Surprise
* Neutral

---

## 2. CK+

Extended Cohn-Kanade Dataset

Used for:

* Facial Action Units
* Expression Analysis
* Model Validation

---

## 3. GoEmotions

Google's large-scale emotion dataset.

Contains:

* 58K+ Reddit comments
* 27 emotion categories
* Fine-grained emotion annotations

Used for:

* Text emotion classification
* DeBERTa fine-tuning

---

# Models Used

## Face Emotion Model

### Primary Model

* FER Library
* MTCNN Face Detection

### Fallback Model

* VGG16 CNN
* Transfer Learning
* Fine-tuned on CK+ and FER2013

Capabilities:

* Face localization
* Expression recognition
* Confidence scoring

---

## Text Emotion Model

### DeBERTa Transformer

Advantages:

* Disentangled Attention Mechanism
* Enhanced Context Understanding
* Better Semantic Representation
* Higher Accuracy than Traditional LSTM Models

Pipeline:

```text
Input Text
    ↓
Tokenizer
    ↓
DeBERTa Encoder
    ↓
Classification Head
    ↓
Emotion + Confidence
```

---

# Fusion Strategy

The project uses a **Confidence-Based Adaptive Fusion Mechanism**.

### Steps

1. Obtain Text Prediction
2. Obtain Face Prediction
3. Calculate Dynamic Weights
4. Normalize Confidence Scores
5. Generate Final Emotion

### Formula

```python
text_weight = 0.3 * (0.5 + text_confidence)

face_weight = 0.7 * (0.5 + face_confidence)

final_confidence =
(text_weight * text_confidence) +
(face_weight * face_confidence)
```

Benefits:

* Better reliability
* Handles uncertain predictions
* Reduces misclassification

---

# Database Design

MongoDB stores:

### Prediction Records

```json
{
  "text": "I am feeling great today",
  "text_emotion": "Happy",
  "face_emotion": "Happy",
  "final_emotion": "Happy",
  "confidence": 0.92,
  "timestamp": "2026-01-15"
}
```

### Stored Data

* User Inputs
* Uploaded Images
* Text Predictions
* Face Predictions
* Fusion Results
* Timestamps

---

# Installation

## Clone Repository

```bash
git clone https://github.com/your-username/multimodal-emotion-recognition.git

cd multimodal-emotion-recognition
```

---

## Create Virtual Environment

```bash
python -m venv .venv

source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Configure Environment Variables

Create `.env`

```env
MONGO_URI=your_mongodb_connection_string
DB_NAME=emotion_db
```

---

# Running the Application

### Start Main Application

```bash
python app.py
```

### Start Admin Dashboard

```bash
python admin.py
```

---

# Results

The system successfully:

 Detects facial emotions

 Detects textual emotions

 Combines multimodal predictions

 Stores prediction history

 Supports confidence-aware inference

 Provides administrative monitoring

---

# Academic Contributions

This project demonstrates:

* Multimodal Machine Learning
* Deep Learning-based Emotion Recognition
* CNN-based Computer Vision
* Transformer-based NLP
* Confidence-aware Fusion
* Database-backed AI Systems

---

# Future Enhancements

* Speech Emotion Recognition
* Real-Time Video Emotion Detection
* Vision Transformers (ViT)
* EfficientNet-Based FER Models
* Explainable AI (XAI)
* Cloud Deployment
* Mobile Application Support
* Emotion Analytics Dashboard

---

**Multimodal Emotion Recognition using CNN and DeBERTa**

Artificial Intelligence & Machine Learning.

---

## ⭐ If you found this project useful, please give it a star!
