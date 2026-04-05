"""
Flask Routes
============
All HTTP endpoints for the Emotion Recognition web application.
"""

import os
import json
import time
import logging
from datetime import datetime
from flask import (
    Blueprint, render_template, request,
    jsonify, session, current_app
)

from audio_utils import save_uploaded_audio, convert_to_wav
from prediction import analyze_audio, load_model_artifacts

logger = logging.getLogger(__name__)

# Prediction history stored in memory (use Redis for production)
_prediction_history = []


def register_routes(app):
    """Register all blueprints and routes with the Flask app."""
    app.register_blueprint(pages_bp)
    app.register_blueprint(api_bp)

    # Load model on startup
    with app.app_context():
        load_model_artifacts(
            app.config["MODEL_PATH"],
            app.config["SCALER_PATH"],
            app.config["LABEL_ENCODER_PATH"],
        )


# ─────────────────────────────────────────────
#  PAGE ROUTES
# ─────────────────────────────────────────────

pages_bp = Blueprint("pages", __name__)


@pages_bp.route("/")
def index():
    """Main dashboard / landing page."""
    return render_template("index.html")


@pages_bp.route("/upload")
def upload():
    """Audio file upload prediction page."""
    return render_template("upload.html")


@pages_bp.route("/realtime")
def realtime():
    """Real-time microphone detection page."""
    return render_template("realtime.html")


# ─────────────────────────────────────────────
#  API ROUTES
# ─────────────────────────────────────────────

api_bp = Blueprint("api", __name__, url_prefix="/api")


@api_bp.route("/predict/upload", methods=["POST"])
def predict_upload():
    """
    POST /api/predict/upload
    Upload an audio file and receive emotion prediction.

    Request: multipart/form-data with 'audio' file field
    Response: JSON with full prediction results
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    file = request.files["audio"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Validate extension
    allowed = current_app.config["ALLOWED_EXTENSIONS"]
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in allowed:
        return jsonify({
            "error": f"File type '{ext}' not allowed. Use: {', '.join(allowed)}"
        }), 400

    # Save uploaded file
    upload_folder = current_app.config["UPLOAD_FOLDER"]
    saved_path = save_uploaded_audio(file, upload_folder)

    if not saved_path:
        return jsonify({"error": "Failed to save uploaded file"}), 500

    try:
        # Convert to WAV if needed
        wav_path = convert_to_wav(saved_path)

        # Run full analysis
        start_time = time.time()
        result = analyze_audio(wav_path, current_app.config)
        inference_ms = round((time.time() - start_time) * 1000, 1)

        if "error" in result:
            return jsonify(result), 422

        # Add metadata
        result["filename"] = file.filename
        result["inference_ms"] = inference_ms
        result["timestamp"] = datetime.now().isoformat()

        # Save to history
        _add_to_history(result)

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        # Clean up uploaded file
        try:
            if os.path.exists(saved_path):
                os.remove(saved_path)
            if wav_path != saved_path and os.path.exists(wav_path):
                os.remove(wav_path)
        except Exception:
            pass


@api_bp.route("/predict/realtime", methods=["POST"])
def predict_realtime():
    """
    POST /api/predict/realtime
    Accept base64-encoded audio blob from browser MediaRecorder.

    Request: JSON {"audio_blob": "<base64>", "mime_type": "audio/webm"}
    Response: JSON with prediction results
    """
    data = request.get_json(silent=True)
    if not data or "audio_blob" not in data:
        return jsonify({"error": "No audio data received"}), 400

    import base64
    import tempfile

    try:
        audio_bytes = base64.b64decode(data["audio_blob"])

        # Write to temp file
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        # Convert to WAV
        wav_path = convert_to_wav(tmp_path)

        # Run analysis
        start_time = time.time()
        result = analyze_audio(wav_path, current_app.config)
        inference_ms = round((time.time() - start_time) * 1000, 1)

        result["inference_ms"] = inference_ms
        result["timestamp"] = datetime.now().isoformat()
        result["source"] = "microphone"

        _add_to_history(result)

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Realtime prediction error: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        try:
            if "tmp_path" in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
            if "wav_path" in locals() and wav_path != tmp_path and os.path.exists(wav_path):
                os.remove(wav_path)
        except Exception:
            pass


@api_bp.route("/history", methods=["GET"])
def get_history():
    """
    GET /api/history
    Returns the last N predictions.
    """
    return jsonify({
        "history": _prediction_history[-20:],
        "count": len(_prediction_history)
    }), 200


@api_bp.route("/history/clear", methods=["DELETE"])
def clear_history():
    """DELETE /api/history/clear — Clear prediction history."""
    _prediction_history.clear()
    return jsonify({"message": "History cleared"}), 200


@api_bp.route("/health", methods=["GET"])
def health():
    """GET /api/health — Health check endpoint for monitoring."""
    return jsonify({
        "status": "ok",
        "version": "1.0.0",
        "service": "Emotion Recognition API",
        "timestamp": datetime.now().isoformat()
    }), 200


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def _add_to_history(result: dict):
    """Add a prediction result to the in-memory history buffer."""
    history_item = {
        "emotion": result.get("emotion"),
        "emoji": result.get("emotion_emoji"),
        "confidence": result.get("confidence"),
        "gender": result.get("gender"),
        "age_group": result.get("age_group"),
        "timestamp": result.get("timestamp"),
        "filename": result.get("filename", "microphone"),
        "source": result.get("source", "upload"),
    }
    _prediction_history.append(history_item)

    # Keep only last MAX_HISTORY entries
    if len(_prediction_history) > 50:
        _prediction_history.pop(0)
