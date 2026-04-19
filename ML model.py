import argparse
import json
import os
import pickle
from pathlib import Path

import librosa
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

try:
    import kagglehub
except Exception:
    kagglehub = None


EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgusted",
    "08": "surprised",
}

ALLOWED_EMOTIONS = ["neutral", "happy", "sad", "angry"]
SAMPLE_RATE = 22050
CHUNK_DURATION = 2.5
OFFSET = 0.4

MODEL_PATH = "speech_emotion_model.pkl"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "label_encoder.pkl"
SUMMARY_PATH = "results.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Train robust SVM for speech emotion recognition.")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="",
        help="Local path to RAVDESS dataset. If not provided, kagglehub download is used.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip augmentation for faster training.",
    )
    return parser.parse_args()


def fix_length(audio, sample_rate, target_duration=CHUNK_DURATION):
    target_len = int(target_duration * sample_rate)
    audio = np.asarray(audio, dtype=np.float32)

    if len(audio) < target_len:
        return np.pad(audio, (0, target_len - len(audio)))
    if len(audio) > target_len:
        return audio[:target_len]
    return audio


def extract_feature(audio, sample_rate):
    audio = fix_length(audio, sample_rate)

    hop_length = 256
    n_fft = 1024

    stft = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate, hop_length=hop_length)
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=64,
    )
    contrast = librosa.feature.spectral_contrast(S=stft, sr=sample_rate)
    zcr = librosa.feature.zero_crossing_rate(audio, hop_length=hop_length)
    rms = librosa.feature.rms(S=stft, hop_length=hop_length)

    parts = []
    for matrix in (mfcc, chroma, mel, contrast, zcr, rms):
        parts.append(np.mean(matrix, axis=1))
        parts.append(np.std(matrix, axis=1))

    return np.hstack(parts).astype(np.float32)


def load_audio(path, sample_rate=SAMPLE_RATE, duration=CHUNK_DURATION, offset=OFFSET):
    audio, _ = librosa.load(path, sr=sample_rate, duration=duration, offset=offset)
    return fix_length(audio, sample_rate, duration)


def add_noise(audio, level=0.003):
    return audio + level * np.random.randn(len(audio))


def add_pitch(audio, sr, steps):
    changed = librosa.effects.pitch_shift(audio, sr=sr, n_steps=steps)
    return fix_length(changed, sr)


def add_stretch(audio, rate):
    stretched = librosa.effects.time_stretch(audio, rate=rate)
    return fix_length(stretched, SAMPLE_RATE)


def resolve_dataset_path(user_path):
    if user_path:
        return user_path

    if kagglehub is None:
        raise RuntimeError("kagglehub is unavailable. Please pass --dataset-path.")

    return kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")


def collect_records(dataset_path):
    records = []
    for wav_path in sorted(Path(dataset_path).rglob("*.wav")):
        parts = wav_path.stem.split("-")
        if len(parts) < 7:
            continue

        emotion_code = parts[2]
        actor = parts[-1]
        emotion = EMOTION_MAP.get(emotion_code)

        if emotion not in ALLOWED_EMOTIONS:
            continue

        records.append({"path": str(wav_path), "emotion": emotion, "actor": actor})

    if not records:
        raise RuntimeError("No valid wav files found. Check dataset path.")

    return records


def build_dataset(records, label_encoder, augment=False):
    X = []
    y = []

    for rec in records:
        audio = load_audio(rec["path"])
        target = label_encoder.transform([rec["emotion"]])[0]

        X.append(extract_feature(audio, SAMPLE_RATE))
        y.append(target)

        if not augment:
            continue

        for aug_audio in (
            add_noise(audio),
            add_pitch(audio, SAMPLE_RATE, 1.5),
            add_pitch(audio, SAMPLE_RATE, -1.5),
            add_stretch(audio, 0.95),
        ):
            X.append(extract_feature(aug_audio, SAMPLE_RATE))
            y.append(target)

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int32)


def main():
    args = parse_args()

    dataset_path = resolve_dataset_path(args.dataset_path)
    print("Dataset path:", dataset_path)

    records = collect_records(dataset_path)
    labels = np.array([r["emotion"] for r in records])
    groups = np.array([r["actor"] for r in records])

    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    encoded_labels = label_encoder.transform(labels)

    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(split.split(np.zeros(len(records)), encoded_labels, groups=groups))

    train_records = [records[i] for i in train_idx]
    test_records = [records[i] for i in test_idx]

    print("Train size:", len(train_records))
    print("Test size:", len(test_records))

    X_train_raw, y_train = build_dataset(
        train_records,
        label_encoder,
        augment=not args.fast,
    )
    X_test_raw, y_test = build_dataset(test_records, label_encoder, augment=False)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    model = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        class_weight="balanced",
        probability=True,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    recall = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    report = classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0,
    )

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Weighted:", f1_weighted)
    print("F1 Macro:", f1_macro)
    print("\nClassification Report:\n")
    print(report)
    print("Classes:", label_encoder.classes_)

    pickle.dump(model, open(MODEL_PATH, "wb"))
    pickle.dump(label_encoder, open(ENCODER_PATH, "wb"))
    pickle.dump(scaler, open(SCALER_PATH, "wb"))

    summary = {
        "model": "SVM_RBF",
        "sample_rate": SAMPLE_RATE,
        "chunk_duration": CHUNK_DURATION,
        "allowed_emotions": ALLOWED_EMOTIONS,
        "metrics": {
            "accuracy": accuracy,
            "precision_weighted": precision,
            "recall_weighted": recall,
            "f1_weighted": f1_weighted,
            "f1_macro": f1_macro,
        },
        "classes": list(label_encoder.classes_),
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved model artifacts and training summary.")


if __name__ == "__main__":
    main()
