# ==============================
# 🎤 Speech Emotion Detection
# Clean Production Version
# ==============================

import numpy as np
import librosa
import pickle

# ==============================
# 🔹 Load Pre-trained Model
# ==============================

model = pickle.load(open("svm_emotion_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# ==============================
# 🔹 Feature Extraction
# ==============================

def extract_feature(audio, sample_rate):
    # Ensure minimum length
    if len(audio) < sample_rate:
        audio = np.pad(audio, (0, sample_rate - len(audio)))

    # MFCC Features
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # Return numeric feature vector
    return np.hstack((mfcc_mean, mfcc_std)).astype(np.float32)

# ==============================
# 🔹 Split Audio into Chunks
# ==============================

def split_audio(audio, sr, chunk_duration=3):
    chunk_sample = int(chunk_duration * sr)
    chunks = []

    for i in range(0, len(audio), chunk_sample):
        chunk = audio[i:i + chunk_sample]

        # Pad small chunks instead of skipping
        if len(chunk) < sr:
            chunk = np.pad(chunk, (0, sr - len(chunk)))

        chunks.append(chunk)

    return chunks

# ==============================
# 🔹 Predict Emotion (Single Chunk)
# ==============================

def predict_emotion(audio, sr):
    feature = extract_feature(audio, sr)

    # Ensure correct datatype
    feature = np.array(feature, dtype=np.float32)

    # Scale features
    feature = scaler.transform([feature])

    # Predict
    prediction = model.predict(feature)

    return label_encoder.inverse_transform(prediction)[0]

# ==============================
# 🔹 Analyze Emotion Timeline
# ==============================

def analyze_emotion(audio, sr, predict_function):
    chunks = split_audio(audio, sr)

    timeline = []

    for i, chunk in enumerate(chunks):
        emotion = predict_function(chunk, sr)

        timeline.append({
            "time": f"{i*3}-{(i+1)*3} sec",
            "emotion": emotion
        })

    return timeline