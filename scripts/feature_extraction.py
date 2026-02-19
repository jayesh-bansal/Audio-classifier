"""
Audio feature extraction utilities for Messy Mashup genre classification.
Supports mel spectrograms and handcrafted features (MFCC, chroma, spectral).
"""

import numpy as np
import librosa


def load_audio(path, sr=22050, duration=10):
    """Load audio file, pad/trim to fixed duration."""
    try:
        y, _ = librosa.load(path, sr=sr, duration=duration, mono=True)
    except Exception:
        y = np.zeros(sr * duration, dtype=np.float32)
    target_len = sr * duration
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y.astype(np.float32)


def audio_to_melspec(y, sr=22050, n_mels=128, n_fft=2048,
                     hop_length=512, fmax=8000, img_width=216):
    """Convert audio to normalized log-mel spectrogram."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft,
        hop_length=hop_length, fmax=fmax
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    if mel_db.shape[1] < img_width:
        mel_db = np.pad(mel_db, ((0, 0), (0, img_width - mel_db.shape[1])))
    else:
        mel_db = mel_db[:, :img_width]
    return mel_db.astype(np.float32)


def extract_handcrafted_features(y, sr=22050):
    """Extract handcrafted audio features for classical ML models."""
    features = []

    # MFCCs (20 coefficients x mean + std = 40)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    features.extend(mfcc.mean(axis=1))
    features.extend(mfcc.std(axis=1))

    # Chroma (12 x 2 = 24)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.extend(chroma.mean(axis=1))
    features.extend(chroma.std(axis=1))

    # Spectral features
    features.append(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
    features.append(librosa.feature.spectral_bandwidth(y=y, sr=sr).mean())
    features.append(librosa.feature.spectral_rolloff(y=y, sr=sr).mean())
    features.append(librosa.feature.spectral_contrast(y=y, sr=sr).mean())

    # Rhythm
    features.append(librosa.feature.tempo(y=y, sr=sr)[0])

    # Zero crossing rate
    features.append(librosa.feature.zero_crossing_rate(y).mean())

    # RMS energy
    features.append(librosa.feature.rms(y=y).mean())

    return np.array(features, dtype=np.float32)
