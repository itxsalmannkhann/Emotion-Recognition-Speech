"""
Real-Time Emotion, Age, Gender and Speech Attribute Recognition from Audio
Main Flask Application Entry Point
Author: AI-Powered Portfolio Project
"""

from flask import Flask
from flask_cors import CORS
from config import Config
from routes import register_routes
import os


def create_app(config_class=Config):
    """
    Application factory pattern.
    Creates and configures the Flask app instance.
    """
    app = Flask(
        __name__,
        template_folder="../templates",
        static_folder="../static"
    )
    app.config.from_object(config_class)

    # Enable CORS for API endpoints
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Register all routes/blueprints
    register_routes(app)

    # Ensure upload directory exists
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    return app


if __name__ == "__main__":
    app = create_app()
    print("🎙️  Emotion Recognition System Starting...")
    print("🌐  Visit: http://localhost:5000")
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
