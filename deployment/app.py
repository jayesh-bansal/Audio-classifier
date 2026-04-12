"""
Messy Mashup - Music Genre Classifier
Gradio app for Hugging Face Spaces deployment.

Upload a .wav file and get genre predictions from an ensemble of models.
"""

import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr

# ============================================================
# Config
# ============================================================
GENRES = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]
IDX2GENRE = {i: g for i, g in enumerate(GENRES)}
SR = 22050
DURATION = 10
N_MELS = 128
IMG_WIDTH = 216

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================
# Audio Processing
# ============================================================
def load_audio(path, sr=SR, duration=DURATION):
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


def audio_to_melspec(y, sr=SR):
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, n_fft=2048, hop_length=512, fmax=8000
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    if mel_db.shape[1] < IMG_WIDTH:
        mel_db = np.pad(mel_db, ((0, 0), (0, IMG_WIDTH - mel_db.shape[1])))
    else:
        mel_db = mel_db[:, :IMG_WIDTH]
    return mel_db.astype(np.float32)


# ============================================================
# Model Definitions (must match training exactly)
# ============================================================
class GenreClassifier(nn.Module):
    """4-block CNN for genre classification."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.3),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.view(x.size(0), -1))


class CRNN(nn.Module):
    """CNN + Bidirectional GRU."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d((2, 2)), nn.Dropout(0.2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d((2, 2)), nn.Dropout(0.2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d((2, 1)), nn.Dropout(0.3),
        )
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))
        self.gru = nn.GRU(128, 128, num_layers=2, batch_first=True,
                          dropout=0.3, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.freq_pool(x).squeeze(2).permute(0, 2, 1)
        out, _ = self.gru(x)
        return self.classifier(out[:, -1, :])


# ============================================================
# Load Models
# ============================================================
MODEL_DIR = os.path.join(os.path.dirname(__file__), "weights")

def load_models():
    """Load CNN and CRNN fold models from weights/ directory."""
    models = []

    # Load CNN folds
    for i in range(5):
        path = os.path.join(MODEL_DIR, f"cnn_fold{i}.pth")
        if os.path.exists(path):
            m = GenreClassifier(10).to(device)
            m.load_state_dict(torch.load(path, map_location=device))
            m.eval()
            models.append(("CNN", m))

    # Load CRNN folds
    for i in range(5):
        path = os.path.join(MODEL_DIR, f"crnn_fold{i}.pth")
        if os.path.exists(path):
            m = CRNN(10).to(device)
            m.load_state_dict(torch.load(path, map_location=device))
            m.eval()
            models.append(("CRNN", m))

    if not models:
        print("WARNING: No model weights found in weights/ directory!")
        print("Place your .pth files in deployment/weights/")

    return models


MODELS = load_models()


# ============================================================
# Prediction
# ============================================================
@torch.no_grad()
def predict(audio_path):
    """
    Predict genre from an audio file.
    Returns a dictionary of genre -> confidence for Gradio's Label output.
    """
    if audio_path is None:
        return {g: 0.0 for g in GENRES}

    # Handle Gradio's (sr, array) tuple format
    if isinstance(audio_path, tuple):
        sr_in, audio_array = audio_path
        # Convert to float32 and normalize
        audio_array = audio_array.astype(np.float32)
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)  # mono
        if np.max(np.abs(audio_array)) > 0:
            audio_array = audio_array / np.max(np.abs(audio_array))
        # Resample if needed
        if sr_in != SR:
            audio_array = librosa.resample(audio_array, orig_sr=sr_in, target_sr=SR)
        target_len = SR * DURATION
        if len(audio_array) < target_len:
            audio_array = np.pad(audio_array, (0, target_len - len(audio_array)))
        else:
            audio_array = audio_array[:target_len]
        y = audio_array.astype(np.float32)
    else:
        y = load_audio(audio_path)

    # Mel spectrogram
    spec = audio_to_melspec(y)
    x = torch.FloatTensor(spec).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, n_mels, time)

    if not MODELS:
        return {g: 1.0 / len(GENRES) for g in GENRES}

    # Ensemble prediction
    all_probs = []
    for name, model in MODELS:
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        all_probs.append(probs)

    avg_probs = np.mean(all_probs, axis=0)

    # Return as {genre: confidence} dict
    return {GENRES[i]: float(avg_probs[i]) for i in range(len(GENRES))}


# ============================================================
# Gradio Interface
# ============================================================
demo = gr.Interface(
    fn=predict,
    inputs=gr.Audio(
        label="Upload a music clip (.wav)",
        type="numpy"
    ),
    outputs=gr.Label(
        num_top_classes=10,
        label="Genre Prediction"
    ),
    title="Messy Mashup - Music Genre Classifier",
    description=(
        "Upload a music clip and the model will predict its genre. "
        "Uses an ensemble of CNN and CRNN models trained on mel spectrograms. "
        "Supports: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock."
    ),
    examples=[],
    theme=gr.themes.Soft(),
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
