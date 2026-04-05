"""
train_demo_model.py
===================
Creates a small demo LSTM model with random weights so the web app
can be tested WITHOUT downloading the full RAVDESS dataset.

This is for DEMO PURPOSES ONLY. For real performance, train using the
Jupyter notebook with the actual RAVDESS dataset.

Usage:
    cd emotion-recognition-speech
    python train_demo_model.py
"""

import os
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

print("🤖 Creating demo LSTM model for testing...")

# Check TensorFlow availability
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
    print(f"✅ TensorFlow {tf.__version__} found")
except ImportError:
    TF_AVAILABLE = False
    print("⚠️  TensorFlow not found — install it with: pip install tensorflow")

if not TF_AVAILABLE:
    print("\nPlease install TensorFlow to create the model:")
    print("  pip install tensorflow")
    exit(1)

# Configuration
NUM_FEATURES = 162
NUM_CLASSES  = 8
MODELS_DIR   = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

EMOTION_LABELS = [
    "angry", "calm", "disgust", "fearful",
    "happy", "neutral", "sad", "surprised"
]


def build_model():
    """Build the LSTM architecture."""
    inputs = keras.Input(shape=(1, NUM_FEATURES), name="audio_features")

    x = layers.LSTM(256, return_sequences=True)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.LSTM(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="emotion_output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="VoiceIQ_LSTM_Demo")
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ── Build model ─────────────────────────────────
model = build_model()
model.summary()

# ── Save model ───────────────────────────────────
model_path = os.path.join(MODELS_DIR, "lstm_model.h5")
model.save(model_path)
print(f"\n✅ Demo model saved: {model_path}")

# ── Create & save a dummy scaler ─────────────────
scaler = StandardScaler()
dummy_X = np.random.randn(100, NUM_FEATURES)
scaler.fit(dummy_X)

scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)
print(f"✅ Demo scaler saved: {scaler_path}")

# ── Create & save label encoder ──────────────────
le = LabelEncoder()
le.fit(EMOTION_LABELS)

le_path = os.path.join(MODELS_DIR, "label_encoder.pkl")
with open(le_path, "wb") as f:
    pickle.dump(le, f)
print(f"✅ Label encoder saved: {le_path}")

print("\n" + "="*55)
print("🎉 Demo model ready!")
print("="*55)
print("\n⚠️  NOTE: This demo model has RANDOM WEIGHTS.")
print("   Predictions will not be accurate.")
print("   Train the real model using the Jupyter notebook")
print("   with the RAVDESS dataset for actual results.\n")
print("▶  Now start the web app:")
print("   cd app && python app.py\n")
