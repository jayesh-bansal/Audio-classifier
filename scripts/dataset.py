"""
Dataset classes and synthetic mashup generation for Messy Mashup competition.
"""

import random
import numpy as np
import torch
from torch.utils.data import Dataset
import librosa

from feature_extraction import load_audio


class MelDataset(Dataset):
    """Dataset for mel spectrogram inputs (used by CNN and CRNN)."""

    def __init__(self, specs, labels=None, augment=False):
        self.specs = specs
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, idx):
        spec = self.specs[idx].copy()

        if self.augment:
            # Frequency masking
            if random.random() < 0.5:
                f = random.randint(0, 20)
                f0 = random.randint(0, max(0, spec.shape[0] - f))
                spec[f0:f0 + f, :] = 0
            # Time masking
            if random.random() < 0.5:
                t = random.randint(0, 30)
                t0 = random.randint(0, max(0, spec.shape[1] - t))
                spec[:, t0:t0 + t] = 0
            # Random gain
            if random.random() < 0.3:
                spec = spec * np.random.uniform(0.8, 1.2)
                spec = np.clip(spec, 0, 1)

        x = torch.FloatTensor(spec).unsqueeze(0)  # (1, n_mels, time)

        if self.labels is not None:
            return x, torch.LongTensor([self.labels[idx]])[0]
        return x


class ASTDataset(Dataset):
    """Dataset for Audio Spectrogram Transformer inputs."""

    def __init__(self, audios, labels=None, feature_extractor=None, sr=22050):
        self.audios = audios
        self.labels = labels
        self.feature_extractor = feature_extractor
        self.sr = sr

    def __len__(self):
        return len(self.audios)

    def __getitem__(self, idx):
        inputs = self.feature_extractor(
            self.audios[idx], sampling_rate=self.sr,
            return_tensors="pt", padding="max_length",
            max_length=1024, truncation=True
        )
        pixel_values = inputs['input_values'].squeeze(0)
        if self.labels is not None:
            return pixel_values, torch.tensor(self.labels[idx], dtype=torch.long)
        return pixel_values


# --- Synthetic mashup generation ---

def load_stem(song_dir, stem_name, sr=22050, duration=10):
    """Load a single stem from a song directory."""
    path = song_dir / stem_name
    if path.exists():
        return load_audio(str(path), sr=sr, duration=duration)
    return np.zeros(sr * duration, dtype=np.float32)


def random_tempo_shift(y, sr, factor=None):
    """Apply random tempo change without pitch shift."""
    if factor is None:
        factor = np.random.uniform(0.85, 1.15)
    return librosa.effects.time_stretch(y, rate=factor)


def add_noise(y, noise_files, sr=22050, duration=10, max_snr=15):
    """Add random noise at random position with random SNR."""
    if len(noise_files) == 0:
        return y
    noise_path = random.choice(noise_files)
    noise = load_audio(str(noise_path), sr=sr, duration=duration)

    snr_db = np.random.uniform(3, max_snr)
    sig_power = np.mean(y ** 2) + 1e-10
    noise_power = np.mean(noise ** 2) + 1e-10
    scale = np.sqrt(sig_power / (noise_power * (10 ** (snr_db / 10))))

    max_offset = max(0, len(y) - len(noise))
    offset = random.randint(0, max_offset) if max_offset > 0 else 0
    y_out = y.copy()
    end = min(offset + len(noise), len(y_out))
    y_out[offset:end] += scale * noise[:end - offset]
    return y_out


def create_synthetic_mashup(genre, stem_catalog, noise_files,
                             sr=22050, duration=10, noise_prob=0.7):
    """Create a mashup by mixing stems from different songs of the same genre."""
    songs = stem_catalog[genre]
    stem_names = ['drums.wav', 'vocals.wav', 'bass.wav', 'others.wav']
    selected_songs = random.choices(songs, k=len(stem_names))
    target_len = sr * duration
    mashup = np.zeros(target_len, dtype=np.float32)

    n_stems = random.randint(2, 4)
    chosen_stems = random.sample(list(zip(stem_names, selected_songs)), n_stems)

    for stem_name, song_dir in chosen_stems:
        stem = load_stem(song_dir, stem_name, sr=sr, duration=duration)
        if random.random() < 0.4:
            try:
                stem = random_tempo_shift(stem, sr)
                if len(stem) < target_len:
                    stem = np.pad(stem, (0, target_len - len(stem)))
                else:
                    stem = stem[:target_len]
            except Exception:
                pass
        gain = np.random.uniform(0.5, 1.5)
        mashup += gain * stem

    peak = np.max(np.abs(mashup)) + 1e-8
    mashup = mashup / peak

    if random.random() < noise_prob:
        for _ in range(random.randint(1, 3)):
            mashup = add_noise(mashup, noise_files, sr=sr, duration=duration)
        peak = np.max(np.abs(mashup)) + 1e-8
        mashup = mashup / peak

    return mashup
