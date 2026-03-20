"""
Inference pipeline for Messy Mashup genre classification.
Loads saved models and generates submission.csv from test mashups.
"""

import numpy as np
import pandas as pd
import torch
import joblib
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from feature_extraction import load_audio, audio_to_melspec, extract_handcrafted_features
from dataset import MelDataset, ASTDataset
from models import GenreClassifier, CRNN, ASTClassifier
from train import get_ensemble_probs


# --- Config ---
GENRES = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]
IDX2GENRE = {i: g for i, g in enumerate(GENRES)}

SR = 22050
DURATION = 10
N_MELS = 128
IMG_WIDTH = 216
BATCH_SIZE = 64
N_FOLDS = 5

# Ensemble weights
W_CNN = 0.25
W_CRNN = 0.20
W_AST = 0.40
W_XGB = 0.15


def run_inference(data_path, model_path, ast_pretrained_path=None):
    """
    Run full inference pipeline.

    Args:
        data_path: Path to competition data (contains test.csv, mashups/)
        model_path: Path to saved model weights (.pth, .pkl)
        ast_pretrained_path: Path to saved AST pretrained weights (optional, for offline)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    data_path = Path(data_path)
    model_path = Path(model_path)

    # Load test metadata
    test_df = pd.read_csv(data_path / 'test.csv')
    mashups_path = data_path / 'mashups'

    # Detect filename pattern
    if 'filename' in test_df.columns:
        test_files = [mashups_path / row['filename'] for _, row in test_df.iterrows()]
    else:
        sample_id = test_df['id'].iloc[0]
        if (mashups_path / f"{sample_id}.wav").exists():
            test_files = [mashups_path / f"{row['id']}.wav" for _, row in test_df.iterrows()]
        elif (mashups_path / f"mashup_{sample_id}.wav").exists():
            test_files = [mashups_path / f"mashup_{row['id']}.wav" for _, row in test_df.iterrows()]
        else:
            all_wavs = sorted(mashups_path.glob('*.wav'))
            test_files = all_wavs[:len(test_df)]

    print(f"Test files: {len(test_files)}")

    # Load audio
    print("Loading test audio...")
    test_audios = np.array([load_audio(str(f), SR, DURATION) for f in tqdm(test_files)])

    # Mel spectrograms
    print("Computing mel spectrograms...")
    test_specs = np.array([audio_to_melspec(y, SR, N_MELS, img_width=IMG_WIDTH)
                           for y in tqdm(test_audios)])

    # Handcrafted features
    print("Extracting handcrafted features...")
    X_test_feat = np.array([extract_handcrafted_features(y, SR) for y in tqdm(test_audios)])
    scaler = joblib.load(model_path / 'scaler.pkl')
    X_test_scaled = scaler.transform(X_test_feat)

    # Mel loader (shared by CNN and CRNN)
    test_mel_ds = MelDataset(test_specs, labels=None, augment=False)
    test_mel_loader = DataLoader(test_mel_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 1. CNN
    print("Loading CNN models...")
    cnn_models = []
    for i in range(N_FOLDS):
        m = GenreClassifier(num_classes=10).to(device)
        m.load_state_dict(torch.load(model_path / f'cnn_fold{i}.pth', map_location=device))
        m.eval()
        cnn_models.append(m)
    cnn_probs = get_ensemble_probs(cnn_models, test_mel_loader, device)

    # 2. CRNN
    print("Loading CRNN models...")
    crnn_models = []
    for i in range(N_FOLDS):
        m = CRNN(num_classes=10).to(device)
        m.load_state_dict(torch.load(model_path / f'crnn_fold{i}.pth', map_location=device))
        m.eval()
        crnn_models.append(m)
    crnn_probs = get_ensemble_probs(crnn_models, test_mel_loader, device)

    # 3. AST
    print("Loading AST model...")
    from transformers import ASTFeatureExtractor
    extractor_path = ast_pretrained_path or "MIT/ast-finetuned-audioset-10-10-0.4593"
    ast_extractor = ASTFeatureExtractor.from_pretrained(extractor_path)

    ast_model = ASTClassifier(num_classes=10, model_name=extractor_path).to(device)
    ast_model.load_state_dict(torch.load(model_path / 'ast_best.pth', map_location=device))
    ast_model.eval()

    ast_test_ds = ASTDataset(test_audios, labels=None, feature_extractor=ast_extractor)
    ast_test_loader = DataLoader(ast_test_ds, batch_size=32, shuffle=False, num_workers=2)
    ast_probs = get_ensemble_probs([ast_model], ast_test_loader, device)

    # 4. XGBoost
    print("Loading XGBoost model...")
    xgb_model = joblib.load(model_path / 'xgb_model.pkl')
    xgb_probs = xgb_model.predict_proba(X_test_scaled)

    # Weighted ensemble
    final_probs = (W_CNN * cnn_probs + W_CRNN * crnn_probs +
                   W_AST * ast_probs + W_XGB * xgb_probs)
    final_preds = final_probs.argmax(axis=1)
    predicted_genres = [IDX2GENRE[p] for p in final_preds]

    # Submission
    submission = pd.DataFrame({'id': test_df['id'], 'genre': predicted_genres})
    submission.to_csv('submission.csv', index=False)
    print(f"\nSubmission saved: {len(submission)} rows")
    print(submission['genre'].value_counts().to_string())

    return submission


if __name__ == '__main__':
    run_inference(
        data_path='/kaggle/input/messy-mashup',
        model_path='/kaggle/input/messy-mashup-models',
        ast_pretrained_path='/kaggle/input/ast-pretrained'
    )
