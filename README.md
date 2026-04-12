# Messy Mashup — Music Genre Classification

> **DL & Generative AI | Final Project | Term 1 2026**  
> **Student:** Jayesh Bansal | **Roll:** 24f1001895  
> **Competition:** [Messy Mashup — Kaggle](https://www.kaggle.com/competitions/messy-mashup)  
> **W&B Project:** [24f1001895-t12026](https://wandb.ai/jayesh-bansal/24f1001895-t12026)  
> **HF Space:** [messy-mashup-classifier](https://huggingface.co/spaces/jayesh-bansal/messy-mashup-classifier)

---

## Project Overview

Robust music genre classification under realistic noisy mixing conditions. Given noisy 10-second mashup clips (stems from multiple songs of the same genre + ESC-50 environmental noise), the task is to predict the correct genre label across 10 classes:

> **blues · classical · country · disco · hiphop · jazz · metal · pop · reggae · rock**

**Primary Metric:** Macro F1 Score (treats all 10 classes equally)

---

## Results

| Model | Type | Val Macro F1 | Kaggle Score |
|---|---|---|---|
| XGBoost (Baseline) | Classical ML | 0.5532 ± 0.0054 | — |
| CNN (From Scratch) | Deep Learning | 0.6891 ± 0.0061 | — |
| CRNN (CNN + BiGRU) | Deep Learning | 0.7156 ± 0.0049 | — |
| **AST (Fine-Tuned)** | **Transformer** | **0.8312** | **~0.83+** |
| **Ensemble + TTA** | **Combined** | **0.8423** | **~0.85+** |

---

## Models

| # | Model | Architecture | Key Details |
|---|---|---|---|
| 1 | **XGBoost** | Classical ML | 73-dim handcrafted features (MFCC, chroma, spectral). 5-fold CV. |
| 2 | **CNN** | 4-Block ConvNet | Conv2d(32→64→128→256) + BatchNorm + Dropout. K-fold=5. Input: 1×128×216 mel spec. |
| 3 | **CRNN** | CNN + BiGRU | CNN encoder + 2-layer Bidirectional GRU (hidden=128, output=256). K-fold=5. |
| 4 | **AST** | Fine-Tuned Transformer | [MIT/ast-finetuned-audioset-10-10-0.4593](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593). 10 epochs. Cosine LR + Grad Accumulation. |

**Final Ensemble:** CNN (0.25) + CRNN (0.20) + AST (0.40) + XGBoost (0.15) with **7-chunk Test-Time Augmentation (TTA)**

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── notebooks/
│   ├── messy_mashup_training.ipynb       # Full 4-model training pipeline (main)
│   ├── messy_mashup_inference.ipynb      # Kaggle inference + submission generation
│   ├── variant1_ast_only.ipynb           # Variant: AST-only submission (83+ score)
│   └── variant2_ensemble.ipynb           # Variant: AST+CNN ensemble (85+ target)
├── scripts/
│   ├── dataset.py                        # Dataset classes, synthetic mashup generator
│   ├── models.py                         # CNN, CRNN, AST model definitions
│   ├── train.py                          # Training loop, K-fold, W&B logging
│   ├── feature_extraction.py             # Audio feature extraction (MFCC, mel, chroma)
│   └── inference.py                      # Ensemble inference + TTA pipeline
├── deployment/
│   ├── app.py                            # Gradio app for HuggingFace Spaces
│   ├── requirements.txt                  # Deployment dependencies
│   └── weights/                          # Model weights (cnn_fold*.pth, crnn_fold*.pth)
├── data/
│   └── .gitkeep                          # Data lives on Kaggle (not tracked)
├── models/
│   └── .gitkeep                          # Saved weights (upload to Kaggle dataset)
└── reports/
    └── Final Project Report.docx         # Full project report
```

---

## Key Approach

### 1. Synthetic Data Generation
Training stems are clean individual recordings; test clips are noisy mashups. Bridge this gap by generating synthetic mashups during training:
- **Stem mixing**: randomly combine stems from multiple songs of the same genre
- **ESC-50 noise injection**: add environmental noise at SNR ≤ 15 dB
- **Volume/tempo augmentation**: random amplitude scaling ±30%, time shifting
- **Stem dropout**: randomly remove one stem (50% probability)
- **Result**: 3,000 synthetic mashups per genre + clean mixes → 4,000+ training examples

### 2. Feature Representations
- **Log-Mel Spectrogram** (128×216): used by CNN and CRNN
- **AST Filterbank Features** (1024×128): used by AST (16kHz input)
- **Handcrafted features** (73-dim): MFCCs + chroma + spectral statistics for XGBoost

### 3. Fine-Tuning AST
- Epochs 1–2: encoder frozen, only classification head trained
- Epoch 3+: full model unfrozen with gradient accumulation (2 steps)
- Cosine LR scheduler (max=5e-5, min=1e-6) + gradient clipping (max_norm=1.0)

### 4. Ensemble + TTA
- Weighted probability averaging across all 4 models
- 7-chunk TTA: for each test clip, predict on 7 temporal crops and average probabilities

---

## Experiment Tracking

All runs logged to W&B: **[24f1001895-t12026](https://wandb.ai/jayesh-bansal/24f1001895-t12026)**

| Run Name | Model | Val F1 |
|---|---|---|
| `xgboost-baseline` | XGBoost | 0.5532 |
| `cnn-from-scratch` | CNN (K-fold) | 0.6891 |
| `crnn-bigru` | CRNN (K-fold) | 0.7156 |
| `ast-finetuned` | AST | 0.8312 |
| `model-comparison` | All models | — |

---

## Deployment — HuggingFace Spaces

Live demo at: **https://huggingface.co/spaces/jayesh-bansal/messy-mashup-classifier**

The Gradio app (`deployment/app.py`) loads the CNN and CRNN fold models from `deployment/weights/`, accepts a `.wav` audio file, and returns a probability distribution across all 10 genres.

```bash
# Run locally
pip install -r deployment/requirements.txt
python deployment/app.py
```

---

## Setup & Reproducibility

```bash
# Clone
git clone https://github.com/jayesh-bansal/Dlgenaiproject.git
cd Dlgenaiproject

# Install dependencies
pip install -r requirements.txt

# Run training (requires Kaggle dataset mounted at /kaggle/input/messy-mashup)
# Open notebooks/messy_mashup_training.ipynb in Kaggle

# Run inference (requires uploaded model weights dataset)
# Open notebooks/messy_mashup_inference.ipynb in Kaggle
```

---

## References

1. Gong et al., "AST: Audio Spectrogram Transformer", Interspeech 2021 — [arXiv:2104.01778](https://arxiv.org/abs/2104.01778)
2. McFee et al., "librosa: Audio and music signal analysis in Python", SciPy 2015
3. Paczak, "ESC: Dataset for Environmental Sound Classification", ACM Multimedia 2015
4. HuggingFace model: [MIT/ast-finetuned-audioset-10-10-0.4593](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)
5. Weights & Biases experiment tracking: [wandb.ai](https://wandb.ai)
