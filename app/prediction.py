"""
Prediction Engine
=================
Handles loading the trained LSTM model and running inference.
Falls back to rule-based heuristics if model file is not found.
"""

import os
import numpy as np
import logging
import pickle
import random
from typing import Dict, Optional

from audio_utils import (
    extract_features,
    extract_speech_attributes,
    estimate_gender_age,
    generate_waveform_data,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────

EMOTION_LABELS = [
    "neutral", "calm", "happy", "sad",
    "angry", "fearful", "disgust", "surprised"
]

EMOTION_EMOJIS = {
    "neutral":   "😐",
    "calm":      "😌",
    "happy":     "😊",
    "sad":       "😢",
    "angry":     "😠",
    "fearful":   "😨",
    "disgust":   "🤢",
    "surprised": "😲",
}

EMOTION_COLORS = {
    "neutral":   "#94a3b8",
    "calm":      "#67e8f9",
    "happy":     "#fbbf24",
    "sad":       "#60a5fa",
    "angry":     "#f87171",
    "fearful":   "#a78bfa",
    "disgust":   "#86efac",
    "surprised": "#fb923c",
}


# ─────────────────────────────────────────────
#  MODEL LOADER (Singleton)
# ─────────────────────────────────────────────

_model = None
_scaler = None
_label_encoder = None


def load_model_artifacts(
    model_path: str,
    scaler_path: str,
    label_encoder_path: str,
) -> bool:
    """
    Load trained Keras model + scaler + label encoder from disk.
    Returns True if successful, False otherwise.
    """
    global _model, _scaler, _label_encoder

    try:
        import tensorflow as tf
        if os.path.exists(model_path):
            _model = tf.keras.models.load_model(model_path)
            logger.info(f"✅ LSTM model loaded from {model_path}")
        else:
            logger.warning(f"⚠️  Model file not found at {model_path}. Using heuristic mode.")
            return False

        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                _scaler = pickle.load(f)
            logger.info("✅ Scaler loaded.")

        if os.path.exists(label_encoder_path):
            with open(label_encoder_path, "rb") as f:
                _label_encoder = pickle.load(f)
            logger.info("✅ Label encoder loaded.")

        return True

    except ImportError:
        logger.warning("TensorFlow not installed. Running in heuristic mode.")
        return False
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        return False


# ─────────────────────────────────────────────
#  PREDICTION FUNCTIONS
# ─────────────────────────────────────────────

def predict_with_model(features: np.ndarray) -> Dict:
    """
    Run inference on the loaded LSTM model.

    Parameters
    ----------
    features : np.ndarray  shape (162,)

    Returns
    -------
    dict with probabilities per emotion class
    """
    global _model, _scaler, _label_encoder

    if _model is None:
        return None

    try:
        # Scale features
        X = features.reshape(1, -1)
        if _scaler is not None:
            X = _scaler.transform(X)

        # Reshape for LSTM input: (samples, timesteps, features)
        X = X.reshape(1, 1, X.shape[1])

        # Run inference
        probabilities = _model.predict(X, verbose=0)[0]

        # Map to label names
        if _label_encoder is not None:
            labels = _label_encoder.classes_
        else:
            labels = EMOTION_LABELS[:len(probabilities)]

        return {label: float(prob) for label, prob in zip(labels, probabilities)}

    except Exception as e:
        logger.error(f"Model inference failed: {e}")
        return None


def predict_heuristic(features: np.ndarray, speech_attrs: Dict) -> Dict:
    """
    Rule-based heuristic predictor when model is unavailable.
    Uses audio features to make educated guesses about emotion.

    This is a DEMO fallback only — production should use the trained model.
    """
    # Extract key indicators
    rms = speech_attrs.get("energy_rms", 0.05)
    pitch = speech_attrs.get("pitch_hz", 150)
    tempo = speech_attrs.get("speech_tempo", 2.5)

    # Build probability distribution based on heuristics
    probs = {label: 0.02 for label in EMOTION_LABELS}  # base probability

    # High energy → more likely angry/happy/surprised
    if rms > 0.1:
        probs["angry"] += 0.35
        probs["happy"] += 0.20
        probs["surprised"] += 0.15

    # Low energy → more likely sad/neutral
    elif rms < 0.03:
        probs["sad"] += 0.35
        probs["neutral"] += 0.25
        probs["calm"] += 0.15

    else:
        probs["neutral"] += 0.25
        probs["calm"] += 0.20
        probs["happy"] += 0.15

    # High pitch variation → more expressive
    if pitch > 250:
        probs["surprised"] += 0.1
        probs["fearful"] += 0.1

    # Fast tempo → excited or angry
    if tempo > 3.5:
        probs["happy"] += 0.15
        probs["angry"] += 0.10
        probs["surprised"] += 0.10

    # Add small random perturbation for realism
    for label in probs:
        probs[label] += random.uniform(0, 0.03)

    # Normalize to sum to 1
    total = sum(probs.values())
    probs = {k: round(v / total, 4) for k, v in probs.items()}

    return probs


def analyze_audio(audio_path: str, config) -> Dict:
    """
    Full analysis pipeline for a given audio file.

    Steps:
    1. Extract acoustic features
    2. Extract speech attributes
    3. Predict emotion (model or heuristic)
    4. Estimate gender and age
    5. Generate waveform data
    6. Build response payload

    Parameters
    ----------
    audio_path : str
    config     : Flask app config object

    Returns
    -------
    dict with all prediction results
    """
    result = {}

    # ── 1. Extract features ───────────────────
    features = extract_features(
        audio_path,
        sample_rate=config.SAMPLE_RATE,
        n_mfcc=config.N_MFCC,
        duration=config.DURATION,
    )

    if features is None:
        return {"error": "Feature extraction failed. Check audio file format."}

    # ── 2. Speech attributes ──────────────────
    speech_attrs = extract_speech_attributes(audio_path, config.SAMPLE_RATE)

    # ── 3. Emotion prediction ─────────────────
    emotion_probs = predict_with_model(features)
    using_model = emotion_probs is not None

    if not using_model:
        emotion_probs = predict_heuristic(features, speech_attrs)

    # Get top emotion
    top_emotion = max(emotion_probs, key=emotion_probs.get)
    top_confidence = round(emotion_probs[top_emotion] * 100, 1)

    # Sort all emotions by probability
    sorted_emotions = sorted(
        [(k, round(v * 100, 1)) for k, v in emotion_probs.items()],
        key=lambda x: x[1],
        reverse=True
    )

    # ── 4. Gender & Age ───────────────────────
    pitch = speech_attrs.get("pitch_hz", 150)
    gender, age_group = estimate_gender_age(pitch)

    # ── 5. Waveform data ──────────────────────
    waveform = generate_waveform_data(audio_path, num_points=150)

    # ── 6. Build response ─────────────────────
    result = {
        "success": True,
        "model_used": using_model,

        # Primary prediction
        "emotion": top_emotion,
        "emotion_emoji": EMOTION_EMOJIS.get(top_emotion, "🎭"),
        "emotion_color": EMOTION_COLORS.get(top_emotion, "#94a3b8"),
        "confidence": top_confidence,

        # All emotion probabilities (for bar chart)
        "emotion_probabilities": sorted_emotions,

        # Gender & Age
        "gender": gender,
        "age_group": age_group,

        # Speech attributes
        "speech_attributes": {
            "energy_level": speech_attrs.get("energy_level", "medium"),
            "energy_rms": speech_attrs.get("energy_rms", 0.0),
            "speech_tempo": speech_attrs.get("speech_tempo", 0.0),
            "confidence_score": speech_attrs.get("confidence", 0.0),
            "duration_sec": speech_attrs.get("duration_sec", 0.0),
            "pitch_hz": speech_attrs.get("pitch_hz", 0.0),
        },

        # Waveform for visualization
        "waveform": waveform,
    }

    return result
