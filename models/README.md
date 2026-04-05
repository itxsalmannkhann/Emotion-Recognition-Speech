# models/

Place your trained model files here after downloading from Google Colab:

- lstm_model.h5        ← Trained Keras LSTM model
- scaler.pkl           ← StandardScaler for feature normalization
- label_encoder.pkl    ← LabelEncoder for emotion class names

## How to get these files

1. Open VoiceIQ_Emotion_Recognition.ipynb in Google Colab
2. Run all cells (takes ~35-50 min with GPU)
3. The last cell downloads all 3 files to your computer
4. Move them into this folder

## Quick test (demo model with random weights)

If you just want to test the UI without training:

    cd ..
    python train_demo_model.py

This creates a demo model instantly (predictions won't be accurate).
