"""
Audio Utility Module
====================
Handles all audio processing and feature extraction for the emotion recognition system.
Uses Librosa for professional-grade signal processing.
"""

import numpy as np
import librosa
import soundfile as sf
import os
import tempfile
import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  FEATURE EXTRACTION
# ─────────────────────────────────────────────

def extract_features(
    audio_path: str,
    sample_rate: int = 22050,
    n_mfcc: int = 40,
    n_chroma: int = 12,
    n_mels: int = 128,
    duration: float = 3.0,
) -> Optional[np.ndarray]:
    """
    Extract a rich feature vector from an audio file.

    Features extracted:
    - MFCC (Mel-Frequency Cepstral Coefficients) - 40 features
    - Chroma features - 12 features
    - Mel Spectrogram - 128 features
    - Zero Crossing Rate - 1 feature
    - Spectral Contrast - 7 features
    - Spectral Rolloff - 1 feature
    - RMS Energy - 1 feature

    Parameters
    ----------
    audio_path : str
        Path to the audio file
    sample_rate : int
        Target sample rate (default 22050 Hz)
    n_mfcc : int
        Number of MFCC coefficients
    n_chroma : int
        Number of chroma bins
    n_mels : int
        Number of mel spectrogram bands
    duration : float
        Max duration in seconds to process

    Returns
    -------
    np.ndarray of shape (190,) or None if extraction fails
    """
    try:
        # Load audio with fixed duration
        y, sr = librosa.load(audio_path, sr=sample_rate, duration=duration)

        # Normalize amplitude
        y = librosa.util.normalize(y)

        # ── 1. MFCC ─────────────────────────────────
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)           # shape: (40,)
        mfcc_std = np.std(mfcc, axis=1)             # shape: (40,)

        # ── 2. Chroma ────────────────────────────────
        stft = np.abs(librosa.stft(y))
        chroma = librosa.feature.chroma_stft(S=stft, sr=sr, n_chroma=n_chroma)
        chroma_mean = np.mean(chroma, axis=1)       # shape: (12,)

        # ── 3. Mel Spectrogram ───────────────────────
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_mean = np.mean(mel, axis=1)             # shape: (128,)

        # ── 4. Zero Crossing Rate ────────────────────
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)                     # scalar

        # ── 5. Spectral Contrast ─────────────────────
        contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)
        contrast_mean = np.mean(contrast, axis=1)   # shape: (7,)

        # ── 6. Spectral Rolloff ──────────────────────
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        rolloff_mean = np.mean(rolloff)             # scalar

        # ── 7. RMS Energy ────────────────────────────
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)                     # scalar

        # ── Concatenate all features ─────────────────
        features = np.concatenate([
            mfcc_mean,          # 40
            mfcc_std,           # 40
            chroma_mean,        # 12
            mel_mean,           # 128  (we'll truncate for efficiency)
            [zcr_mean],         # 1
            contrast_mean,      # 7
            [rolloff_mean],     # 1
            [rms_mean],         # 1
        ])

        # Trim mel to keep vector manageable (first 60 bins)
        features = np.concatenate([
            mfcc_mean,          # 40
            mfcc_std,           # 40
            chroma_mean,        # 12
            mel_mean[:60],      # 60
            [zcr_mean],         # 1
            contrast_mean,      # 7
            [rolloff_mean],     # 1
            [rms_mean],         # 1
        ])
        # Total = 162 features

        return features.astype(np.float32)

    except Exception as e:
        logger.error(f"Feature extraction failed for {audio_path}: {e}")
        return None


def extract_speech_attributes(audio_path: str, sample_rate: int = 22050) -> Dict:
    """
    Extract human-readable speech attributes from audio.

    Returns
    -------
    dict with keys:
        energy_level  : str  (low / medium / high)
        speech_tempo  : float (estimated syllables/sec)
        confidence    : float (0–1, based on signal clarity)
        duration_sec  : float
        pitch_hz      : float (fundamental frequency estimate)
    """
    try:
        y, sr = librosa.load(audio_path, sr=sample_rate)
        y_norm = librosa.util.normalize(y)

        # ── Duration ─────────────────────────────────
        duration_sec = round(len(y) / sr, 2)

        # ── Energy level ─────────────────────────────
        rms = float(np.mean(librosa.feature.rms(y=y_norm)))
        if rms < 0.02:
            energy_level = "low"
        elif rms < 0.08:
            energy_level = "medium"
        else:
            energy_level = "high"

        # ── Speech tempo (beat-based estimate) ───────
        try:
            tempo, _ = librosa.beat.beat_track(y=y_norm, sr=sr)
            speech_tempo = round(float(tempo) / 60.0, 2)  # beats/sec → approx syllables/sec
        except Exception:
            speech_tempo = 2.5  # default

        # ── Pitch estimate via YIN ───────────────────
        try:
            f0 = librosa.yin(y_norm, fmin=50, fmax=400, sr=sr)
            pitch_hz = round(float(np.nanmedian(f0[f0 > 0])), 1)
        except Exception:
            pitch_hz = 0.0

        # ── Confidence (spectral flatness proxy) ─────
        flatness = librosa.feature.spectral_flatness(y=y_norm)
        # Lower flatness = more tonal = more speech-like → higher confidence
        confidence = round(float(1.0 - np.clip(np.mean(flatness) * 10, 0, 1)), 2)

        return {
            "energy_level": energy_level,
            "energy_rms": round(rms, 4),
            "speech_tempo": speech_tempo,
            "confidence": confidence,
            "duration_sec": duration_sec,
            "pitch_hz": pitch_hz,
        }

    except Exception as e:
        logger.error(f"Speech attribute extraction failed: {e}")
        return {
            "energy_level": "unknown",
            "energy_rms": 0.0,
            "speech_tempo": 0.0,
            "confidence": 0.0,
            "duration_sec": 0.0,
            "pitch_hz": 0.0,
        }


def estimate_gender_age(pitch_hz: float) -> Tuple[str, str]:
    """
    Heuristic gender and age estimation from fundamental pitch frequency.

    Male speech:  ~85–180 Hz
    Female speech: ~165–255 Hz
    Children:      ~250–400 Hz

    Parameters
    ----------
    pitch_hz : float  Fundamental frequency in Hz

    Returns
    -------
    (gender: str, age_group: str)
    """
    if pitch_hz <= 0:
        return "unknown", "unknown"

    if pitch_hz < 165:
        gender = "Male"
        if pitch_hz < 100:
            age_group = "Senior (50+)"
        else:
            age_group = "Adult (20-50)"
    elif pitch_hz < 255:
        gender = "Female"
        if pitch_hz > 220:
            age_group = "Teen (13-19)"
        else:
            age_group = "Adult (20-50)"
    else:
        gender = "Child"
        age_group = "Child (0-12)"

    return gender, age_group


def generate_waveform_data(audio_path: str, num_points: int = 200) -> list:
    """
    Generate downsampled waveform data for frontend visualization.

    Parameters
    ----------
    audio_path : str
    num_points : int  Number of data points to return

    Returns
    -------
    list of float values in range [-1, 1]
    """
    try:
        y, sr = librosa.load(audio_path, sr=8000, duration=10.0)
        y = librosa.util.normalize(y)

        # Downsample to num_points
        step = max(1, len(y) // num_points)
        waveform = y[::step][:num_points].tolist()

        return [round(float(v), 4) for v in waveform]
    except Exception as e:
        logger.error(f"Waveform generation failed: {e}")
        return [0.0] * num_points


def save_uploaded_audio(file_storage, upload_folder: str) -> Optional[str]:
    """
    Safely save an uploaded audio file to disk.

    Parameters
    ----------
    file_storage : werkzeug.FileStorage
    upload_folder : str

    Returns
    -------
    str  Path to saved file, or None
    """
    try:
        import uuid
        ext = os.path.splitext(file_storage.filename)[-1].lower()
        unique_name = f"{uuid.uuid4().hex}{ext}"
        save_path = os.path.join(upload_folder, unique_name)
        os.makedirs(upload_folder, exist_ok=True)
        file_storage.save(save_path)
        return save_path
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        return None


def convert_to_wav(audio_path: str) -> str:
    """
    Convert any audio format to WAV using soundfile (if needed).
    Returns path to WAV file.
    """
    if audio_path.endswith(".wav"):
        return audio_path

    try:
        data, samplerate = sf.read(audio_path)
        wav_path = audio_path.rsplit(".", 1)[0] + ".wav"
        sf.write(wav_path, data, samplerate)
        return wav_path
    except Exception as e:
        logger.warning(f"WAV conversion failed, using original: {e}")
        return audio_path
