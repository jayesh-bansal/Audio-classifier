"""
Model architectures for Messy Mashup genre classification.
- GenreClassifier: CNN from scratch
- CRNN: CNN + Bidirectional GRU
- ASTClassifier: Pretrained Audio Spectrogram Transformer
"""

import torch.nn as nn
from transformers import ASTForAudioClassification


class GenreClassifier(nn.Module):
    """4-block CNN for mel spectrogram genre classification."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.2),
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.2),
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.3),
            # Block 4
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class CRNN(nn.Module):
    """CNN + Bidirectional GRU for temporal genre classification."""

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
            nn.Dropout(0.4),
            nn.Linear(256, 64),  # 128*2 for bidirectional
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x)                          # (B, 128, F', T')
        x = self.freq_pool(x)                    # (B, 128, 1, T')
        x = x.squeeze(2).permute(0, 2, 1)        # (B, T', 128)
        out, _ = self.gru(x)                      # (B, T', 256)
        x = out[:, -1, :]                         # last time step
        return self.classifier(x)


class ASTClassifier(nn.Module):
    """Pretrained Audio Spectrogram Transformer for genre classification."""

    def __init__(self, num_classes=10,
                 model_name="MIT/ast-finetuned-audioset-10-10-0.4593"):
        super().__init__()
        self.ast = ASTForAudioClassification.from_pretrained(
            model_name, num_labels=num_classes, ignore_mismatched_sizes=True
        )

    def forward(self, x):
        return self.ast(x).logits
