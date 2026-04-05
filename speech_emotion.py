# Imporing necessary libraries
from json import encoder
from json import encoder
import os
from pyexpat import features
import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Emotion mapping based on RAVDESS dataset

emotion_dict = {
    "01":"neutral",
    "02":"calm",
    "03":"happy",
    "04":"sad",
    "05":"angry",
    "06":"fearful",
    "07":"disgusted",
    "08":"surprised"
}

# Feature extraction function

def extract_feature(audio, sample_rate):

  if len(audio)<sample_rate:
     audio = np.pad(audio, (0, sample_rate - len(audio)))

  mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

  mfcc_mean = np.mean(mfcc, axis=1)
  mfcc_std = np.std(mfcc, axis=1)

  return np.hstack((mfcc_mean, mfcc_std))

def split_audio(audio, sr, chunk_duration=3):
    chunk_sample = int(chunk_duration * sr)

    chunks = []
    for i in range(0, len(audio), chunk_sample):
        chunk = audio[i:i + chunk_sample]

        if len(chunk) < sr:  # Skip chunks shorter than 1 second
            chunks.append(chunk)

    return chunks

def analyze_emotion(audio, sr, predict_function):
    chunks = split_audio(audio, sr)

    timeline = []
    for i, chunk in enumerate(chunks):
        emotion = predict_function(chunk, sr)
        timeline.append({
           "time": f"{i * 3}-{(i+1) * 3} seconds",
            "emotion": emotion
        })

    return timeline

# Load dataset and prepare features and labels

import kagglehub

path = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")

print("Path to dataset files:", path)

X=[]
y=[]

allowed_emotions = ["calm", "happy", "sad", "angry"]

dataset_path = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")

for actor in os.listdir(dataset_path):
  actor_path = os.path.join(dataset_path, actor)

  if not os.path.isdir(actor_path):
    continue

  for file in os.listdir(actor_path):
    if file.endswith(".wav"):
      file_path = os.path.join(actor_path, file)

      #Extract emotion from filename
      emotion_code = file.split("-")[2]
      emotion = emotion_dict[emotion_code]

      if emotion not in allowed_emotions:
        continue

      #Extract MFCC feature
      audio,sample_rate = librosa.load(file_path, duration=6, offset=0.5)
      feature = extract_feature(audio, sample_rate)

      X.append(feature)
      y.append(emotion)

import numpy as np
from sklearn.preprocessing import LabelEncoder

# Create numpy arrays

X = np.array(X)
y = np.array(y)

# Encode Emotion labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

from sklearn.preprocessing import StandardScaler

# Standardize features

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2,random_state=42,stratify=y_encoded)

# Train SVM model

model = SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced')
model.fit(X_train, y_train)

#save the model

import pickle
import numpy as np

pickle.dump(model, open("svm_emotion_model.pkl", "wb"))
pickle.dump(label_encoder, open("label_encoder.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

def predict_emotion(audio, sr):
    feature = extract_feature(audio, sr)
    feature = scaler.transform([features])

    prediction = model.predict(feature)
    return label_encoder.inverse_transform(prediction)[0]