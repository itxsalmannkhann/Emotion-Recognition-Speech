# 🎙️ VoiceIQ — Real-Time Emotion, Age, Gender & Speech Recognition

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12-FF6F00?style=flat-square&logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=flat-square&logo=flask)
![Librosa](https://img.shields.io/badge/Librosa-0.10-8A2BE2?style=flat-square)

**An AI-powered speech analysis system that predicts emotion, gender, age group, and speech attributes from audio — in real time.**

</div>

---

## 🎯 Project Overview

VoiceIQ is a production-ready machine learning application that uses a trained LSTM neural network to analyze speech audio and predict:

| Attribute | Details |
|-----------|---------|
| **Emotion** | neutral, calm, happy, sad, angry, fearful, disgust, surprised |
| **Gender** | Male / Female / Child |
| **Age Group** | Child (0-12), Teen (13-19), Adult (20-50), Senior (50+) |
| **Energy Level** | Low / Medium / High |
| **Speech Tempo** | Syllables per second estimate |
| **Pitch (Hz)** | Fundamental frequency |
| **Confidence Score** | Prediction confidence (0–100%) |

---

## ✨ Features

- 🎤 **Real-time microphone recording** with live waveform visualization
- 📁 **Audio file upload** (WAV, MP3, OGG, FLAC, M4A)
- 📊 **Interactive charts**: emotion probability distribution, waveform
- 🌓 **Dark/Light mode** with persistent preference
- 📜 **Prediction history** panel
- 📥 **JSON report download**
- 🔌 **REST API** for integration with other apps
- 🐳 **Docker ready** for deployment

---

## 🏗️ Project Architecture

```
emotion-recognition-speech/
│
├── app/
│   ├── app.py           # Flask application factory
│   ├── routes.py        # All HTTP endpoints (pages + API)
│   ├── audio_utils.py   # Audio loading & feature extraction
│   ├── prediction.py    # LSTM inference engine
│   └── config.py        # Configuration & constants
│
├── models/
│   ├── lstm_model.h5    # Trained Keras model
│   ├── scaler.pkl       # StandardScaler for features
│   └── label_encoder.pkl
│
├── static/
│   ├── css/main.css     # Full design system (glassmorphism)
│   └── js/
│       ├── main.js      # Shared utilities, theme, history
│       ├── upload.js    # Upload page logic + charts
│       ├── realtime.js  # MediaRecorder + live prediction
│       └── waveform.js  # Animated hero canvas wave
│
├── templates/
│   ├── index.html       # Main dashboard
│   ├── upload.html      # File upload + results
│   └── realtime.html    # Live microphone detection
│
├── notebooks/
│   └── emotion_recognition_experiments.ipynb  # Full ML training notebook
│
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🧠 Model Architecture

```
Input (162 features)
    │
    ▼
LSTM(256 units, return_sequences=True)
    │
BatchNormalization + Dropout(0.3)
    │
    ▼
LSTM(128 units)
    │
BatchNormalization + Dropout(0.3)
    │
    ▼
Dense(128, ReLU) → Dropout(0.3)
    │
Dense(64, ReLU)  → Dropout(0.2)
    │
    ▼
Dense(8, Softmax)   ← Output: 8 emotion classes
```

### Feature Vector (162 dimensions)

| Feature | Dims | Description |
|---------|------|-------------|
| MFCC Mean | 40 | Timbre/texture of speech |
| MFCC Std | 40 | Variation in timbre |
| Chroma | 12 | Tonal/harmonic content |
| Mel Spectrogram | 60 | Frequency band energy |
| Zero Crossing Rate | 1 | Signal noisiness |
| Spectral Contrast | 7 | Peak vs valley in spectrum |
| Spectral Rolloff | 1 | High-frequency energy |
| RMS Energy | 1 | Loudness |

---

## 📦 Installation

### Prerequisites
- Python 3.10+
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/voiceiq-emotion-recognition.git
cd voiceiq-emotion-recognition

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
cd app
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

### Troubleshooting
If you encounter any issues during installation, ensure that your Python version is compatible with the project's requirements. You can check the Python version by running `python --version` in your terminal. Additionally, make sure to activate the virtual environment before installing dependencies. If you are still experiencing issues, try updating your pip version by running `pip install --upgrade pip`. You can also try reinstalling the dependencies by running `pip install -r requirements.txt --force-reinstall`. If none of these steps resolve the issue, you can try checking the project's issue tracker for known issues or seeking help from the project's community. For example, if you encounter a `ModuleNotFoundError`, you can try installing the missing module manually using pip. If you encounter a `PermissionError`, you can try running the command with administrator privileges.

---

## 🐳 Docker Deployment

```bash
# Build the image
docker build -t voiceiq .

# Run the container
docker run -p 5000:5000 voiceiq

# Or with docker-compose
docker-compose up --build
```

---

## 📊 Training the Model

> The model requires the **RAVDESS** dataset. Download it from:
> https://zenodo.org/record/1188976

```bash
# 1. Download RAVDESS dataset and place in:
#    notebooks/ravdess-dataset/

# 2. Open the Jupyter notebook
jupyter notebook notebooks/emotion_recognition_experiments.ipynb

# 3. Run all cells to train and save the model
```

The trained model will be saved to `models/lstm_model.h5`.

---

## 🔌 REST API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/predict/upload` | Predict from uploaded audio file |
| `POST` | `/api/predict/realtime` | Predict from base64 audio blob |
| `GET` | `/api/history` | Get prediction history |
| `DELETE` | `/api/history/clear` | Clear prediction history |
| `GET` | `/api/health` | Health check |

### Example: Upload API

```bash
curl -X POST http://localhost:5000/api/predict/upload \
  -F "audio=@sample.wav"
```

### Example Response

```json
{
  "success": true,
  "emotion": "happy",
  "emotion_emoji": "😊",
  "confidence": 87.3,
  "gender": "Female",
  "age_group": "Adult (20-50)",
  "speech_attributes": {
    "energy_level": "high",
    "speech_tempo": 3.2,
    "pitch_hz": 215.4,
    "duration_sec": 3.0
  },
  "emotion_probabilities": [
    ["happy", 87.3],
    ["surprised", 6.1],
    ["neutral", 3.4]
  ],
  "inference_ms": 142.5
}
```

---

## 📈 Expected Model Performance

| Metric | Target |
|--------|--------|
| Accuracy | ~70-80% (RAVDESS 8-class) |
| Precision | ~75% |
| Recall | ~72% |
| F1 Score | ~73% |

*Performance varies by training data size and augmentation.*

---

## 🔮 Future Improvements

- [ ] Multi-speaker diarization
- [ ] Real-time streaming via WebSockets
- [ ] Export to TensorFlow Lite for mobile
- [ ] Multi-language support
- [ ] Transformer-based model (Wav2Vec 2.0)
- [ ] Confidence calibration
- [ ] CREPE pitch estimation
- [ ] Batch API endpoint

---

## 🏫 About

**Name**: Salman Khan
**University**: Abdul Wali Khan University Mardan  
**Course**: Artificial Intelligence 
**Semester**: 6th Semester
**Project Name**: Emotion Recognition System
**Contributer**: Ihsan Ullah Mohmand
---

## 📄 License

MIT License — feel free to use for educational and portfolio purposes.