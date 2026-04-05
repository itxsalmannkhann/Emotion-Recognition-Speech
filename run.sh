#!/usr/bin/env bash
# VoiceIQ — Quick Start Script
# Usage: bash run.sh

set -e

echo ""
echo "🎙️  VoiceIQ — Emotion Recognition System"
echo "========================================="
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.10+"
    exit 1
fi

PYTHON=python3
echo "✅ Python: $($PYTHON --version)"

# Create virtual environment if missing
if [ ! -d "venv" ]; then
    echo ""
    echo "📦 Creating virtual environment..."
    $PYTHON -m venv venv
fi

# Activate venv
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null

# Install requirements
echo ""
echo "📦 Installing requirements..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Create demo model if no model exists
if [ ! -f "models/lstm_model.h5" ]; then
    echo ""
    echo "🤖 No trained model found — creating demo model..."
    python train_demo_model.py
fi

# Start the app
echo ""
echo "🚀 Starting VoiceIQ..."
echo "🌐 Open: http://localhost:5000"
echo ""
cd app && python app.py
