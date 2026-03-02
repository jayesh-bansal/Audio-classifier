"""
Training and validation utilities for Messy Mashup genre classification.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score


def train_one_epoch(model, loader, optimizer, scheduler, device):
    """Train model for one epoch. Returns average loss and macro F1."""
    model.train()
    total_loss = 0
    preds_all, labels_all = [], []

    for specs, labels in loader:
        specs, labels = specs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(specs)
        loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * len(labels)
        preds_all.extend(logits.argmax(1).cpu().numpy())
        labels_all.extend(labels.cpu().numpy())

    scheduler.step()
    avg_loss = total_loss / len(labels_all)
    f1 = f1_score(labels_all, preds_all, average='macro')
    return avg_loss, f1


@torch.no_grad()
def validate(model, loader, device):
    """Validate model. Returns average loss, macro F1, and predictions."""
    model.eval()
    total_loss = 0
    preds_all, labels_all = [], []

    for specs, labels in loader:
        specs, labels = specs.to(device), labels.to(device)
        logits = model(specs)
        loss = F.cross_entropy(logits, labels)
        total_loss += loss.item() * len(labels)
        preds_all.extend(logits.argmax(1).cpu().numpy())
        labels_all.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(labels_all)
    f1 = f1_score(labels_all, preds_all, average='macro')
    return avg_loss, f1, np.array(preds_all)


@torch.no_grad()
def get_ensemble_probs(models, loader, device):
    """Get averaged softmax probabilities from an ensemble of models."""
    all_probs = []
    for model in models:
        model.eval()
        probs = []
        for batch in loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            logits = model(x)
            probs.append(F.softmax(logits, dim=1).cpu().numpy())
        all_probs.append(np.concatenate(probs, axis=0))
    return np.mean(all_probs, axis=0)
