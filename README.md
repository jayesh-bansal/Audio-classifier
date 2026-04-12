# Messy Mashup — Music Genre Classification

> **DL & Generative AI | Final Project | Term 1 2026**  
> **Student:** Jayesh Bansal | **Roll:** 24f1001895  
> **Competition:** [Messy Mashup — Kaggle](https://www.kaggle.com/competitions/messy-mashup)  
> **W&B Project:** [jayeshbans/24f1001895-t12026](https://wandb.ai/jayeshbans/24f1001895-t12026)  
> **HF Space:** [jayesh-bansal/Audio_Classifier](https://huggingface.co/spaces/jayesh-bansal/Audio_Classifier)  
> **GitHub:** [jayesh-bansal/Dlgenaiproject](https://github.com/jayesh-bansal/Dlgenaiproject)

---

## Problem Statement

10-class music genre classification on the Messy Mashup Kaggle competition. Each test clip is a noisy mashup of stems from the same genre mixed with ESC-50 environmental noise. The model must predict the correct genre across 10 classes:

> **blues · classical · country · disco · hiphop · jazz · metal · pop · reggae · rock**

**Primary Metric:** Macro F1 Score (3020 test clips)

---

## Abstract

| Model | Type | Metric | Score |
|---|---|---|---|
| XGBoost | Classical ML | 5-Fold CV F1 | 0.5673 |
| CNN (StrongCNN) | Deep Learning | OOF F1 | 0.6419 |
| CRNN (CNN + BiGRU) | Deep Learning | OOF F1 | 0.4320 |
| AST (Fine-Tuned) | Transformer | Val Macro F1 | 0.8939 |
| **Final Ensemble (AST×0.75 + CNN×0.25)** | **Combined** | **7-chunk TTA** | **3020 test predictions** |

---

## Dataset & EDA

- **Training set:** 1000 clips (100 per genre), 10-second WAV files at 22050 Hz
- **Test set:** 3020 clips, genres unlabelled
- Classes are perfectly balanced in the training split (100 clips/genre)
- Audio is resampled to **16 kHz** for all deep learning models
- Duration clipped / padded to **5 seconds** (80,000 samples) per chunk

---

## Feature Extraction

| Method | Details | Used By |
|---|---|---|
| Log-Mel Spectrogram | 128 mel bins, normalized per clip (mean/std) | CNN, CRNN |
| AST Filterbank Features | ASTFeatureExtractor (MIT/ast-finetuned-audioset-10-10-0.4593) | AST |
| Handcrafted (73-dim) | MFCCs + chroma + spectral centroid/rolloff/contrast | XGBoost |

Mel spectrogram normalization: `mel = (mel - mel.mean()) / (mel.std() + 1e-6)`

---

## Models

### 1 · XGBoost Baseline

- 73-dimensional handcrafted audio features (MFCCs, chroma, spectral statistics)
- 5-fold stratified cross-validation
- **CV Macro F1: 0.5673**
- W&B run: `xgboost-baseline` (`kpftywsz`)

### 2 · CNN (StrongCNN)

4-block convolutional network trained from scratch on 128-bin mel spectrograms:

```
Conv2d(1→32) → BN → ReLU → Conv2d(32→32) → BN → ReLU → MaxPool2d(2) → Dropout(0.2)
Conv2d(32→64) → BN → ReLU → Conv2d(64→64) → BN → ReLU → MaxPool2d(2) → Dropout(0.2)
Conv2d(64→128) → BN → ReLU → Conv2d(128→128) → BN → ReLU → MaxPool2d(2) → Dropout(0.3)
Conv2d(128→256) → BN → ReLU → AdaptiveAvgPool2d(1,1)
→ Dropout(0.4) → Linear(256→128) → ReLU → Dropout(0.3) → Linear(128→10)
```

- Input: 1×128 mel spectrogram, 5-second windows
- **OOF Macro F1: 0.6419**
- W&B run: `cnn-from-scratch` (`n7r73tn2`)

### 3 · CRNN (CNN + BiGRU)

3-block CNN encoder feeding into a 2-layer Bidirectional GRU:

- CNN encoder: channels 32 → 64 → 128, each block with BN + ReLU + MaxPool + Dropout
- BiGRU: hidden=128, bidirectional → output dim=256
- Final FC: 256 → 10 classes
- **OOF Macro F1: 0.4320**
- W&B run: `crnn-bigru` (`5waa4t5l`)

### 4 · AST (Audio Spectrogram Transformer)

Fine-tuned from `MIT/ast-finetuned-audioset-10-10-0.4593`:

- **Epochs 1–2:** encoder frozen, classification head only
- **Epoch 3+:** full model unfrozen with gradient accumulation (2 steps)
- Cosine LR scheduler (max=5e-5) + gradient clipping (max_norm=1.0)
- **Val Macro F1: 0.8939**
- W&B run: `ast-finetuned` (`vx39aonq`)

---

## Final Ensemble & Submission

The final submission uses a **weighted probability average** of AST and StrongCNN with **7-chunk Test-Time Augmentation (TTA)**:

```python
final = 0.75 * ast_probs + 0.25 * cnn_probs
```

**7-chunk TTA:** For each audio clip, 7 evenly-spaced 5-second windows are extracted, each model predicts on all 7 chunks, and probabilities are averaged before combining the ensemble.

| Model | Weight | Val Macro F1 |
|---|---|---|
| AST (fine-tuned) | 0.75 | 0.8939 |
| StrongCNN | 0.25 | 0.6274 |

---

## Experiment Tracking

All runs logged to W&B: **[jayeshbans/24f1001895-t12026](https://wandb.ai/jayeshbans/24f1001895-t12026)**

| Run Name | Run ID | Model | Metric |
|---|---|---|---|
| `xgboost-baseline` | `kpftywsz` | XGBoost | CV F1 = 0.5673 |
| `cnn-from-scratch` | `n7r73tn2` | StrongCNN | OOF F1 = 0.6419 |
| `crnn-bigru` | `5waa4t5l` | CRNN/BiGRU | OOF F1 = 0.4320 |
| `ast-finetuned` | `vx39aonq` | AST | Val F1 = 0.8939 |
| `model-comparison` | `yxrmq0w8` | All models | — |

---

## Deployment

**Live demo:** [huggingface.co/spaces/jayesh-bansal/Audio_Classifier](https://huggingface.co/spaces/jayesh-bansal/Audio_Classifier)

The Gradio app (`deployment/app.py`) loads `weights/ast_best.pth` and `weights/best_cnn.pth`, accepts a `.wav` or `.mp3` audio file, and returns genre probabilities using the AST×0.75 + CNN×0.25 ensemble with 7-chunk TTA.

```bash
# Run locally
pip install -r deployment/requirements.txt
python deployment/app.py
```

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── notebooks/                             # Training and inference notebooks
├── deployment/
│   ├── app.py                             # Gradio app (AST+CNN ensemble, 7-chunk TTA)
│   ├── requirements.txt                   # Deployment dependencies
│   ├── upload_to_hf.py                    # HuggingFace Spaces upload script
│   └── weights/                           # ast_best.pth, best_cnn.pth
└── Final Project Report.docx             # Full project report
```

---

## References

1. Gong et al., "AST: Audio Spectrogram Transformer", Interspeech 2021 — [arXiv:2104.01778](https://arxiv.org/abs/2104.01778)
2. McFee et al., "librosa: Audio and music signal analysis in Python", SciPy 2015
3. Paczak, "ESC: Dataset for Environmental Sound Classification", ACM Multimedia 2015
4. HuggingFace model: [MIT/ast-finetuned-audioset-10-10-0.4593](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)
5. Weights & Biases experiment tracking: [wandb.ai](https://wandb.ai)
