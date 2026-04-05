"""
Configuration settings for the Emotion Recognition Application.
"""

import os
from datetime import timedelta


class Config:
    """Base configuration class."""

    # Flask core settings
    SECRET_KEY = os.environ.get("SECRET_KEY", "emotion-ai-secret-2024-xk9")
    DEBUG = os.environ.get("DEBUG", "True") == "True"

    # File upload settings
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "..", "uploads")
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB max upload
    ALLOWED_EXTENSIONS = {"wav", "mp3", "ogg", "flac", "m4a"}

    # Model paths
    MODEL_PATH = os.path.join(
        os.path.dirname(__file__), "..", "models", "lstm_model.h5"
    )
    SCALER_PATH = os.path.join(
        os.path.dirname(__file__), "..", "models", "scaler.pkl"
    )
    LABEL_ENCODER_PATH = os.path.join(
        os.path.dirname(__file__), "..", "models", "label_encoder.pkl"
    )

    # Audio processing settings
    SAMPLE_RATE = 22050
    DURATION = 3.0          # seconds per chunk
    N_MFCC = 40             # number of MFCC features
    N_CHROMA = 12           # chroma features
    N_MELS = 128            # mel spectrogram bins
    HOP_LENGTH = 512
    N_FFT = 2048

    # Model architecture settings
    NUM_EMOTIONS = 7        # happy, sad, angry, fear, neutral, surprise, disgust
    NUM_GENDERS = 2         # male, female
    NUM_AGE_GROUPS = 4      # child, young-adult, adult, senior

    # Session/history settings
    MAX_HISTORY = 20        # max predictions to keep in history
    SESSION_LIFETIME = timedelta(hours=24)

    # Emotion labels (aligned with RAVDESS dataset)
    EMOTIONS = {
        "01": "neutral",
        "02": "calm",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fearful",
        "07": "disgust",
        "08": "surprised",
    }

    EMOTION_LABELS = [
        "neutral", "calm", "happy", "sad",
        "angry", "fearful", "disgust", "surprised"
    ]

    AGE_GROUPS = ["Child (0-12)", "Teen (13-19)", "Adult (20-50)", "Senior (50+)"]


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False
    SECRET_KEY = os.environ.get("SECRET_KEY", "change-me-in-production")
