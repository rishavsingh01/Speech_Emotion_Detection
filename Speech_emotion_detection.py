# Imporing necessary libraries

import os
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

def extract_feature(file_path):
  audio, sample_rate = librosa.load(file_path, duration=6, offset=0.5)
  mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
  mfcc_mean = np.mean(mfcc, axis=1)
  mfcc_std = np.std(mfcc, axis=1)

  return np.hstack((mfcc_mean, mfcc_std))

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

      #Extract MFCC features
      feature = extract_feature(file_path)

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

# Evaluate the model 

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy*100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred,target_names=label_encoder.classes_))

# Function to record live audio
import sounddevice as sd
import scipy.io.wavfile as wav

def record_audio(duration=5, fs=44100, filename="live_audio.wav"):
    print("Recording(Start Saying Somthing)...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    wav.write(filename, fs, recording)
    print("Recording complete.")

    # Real time emotion prediction

def predict_emotion_live():
      record_audio()
      y,sr = librosa.load("live_audio.wav", sr=44100)
      y, _ = librosa.effects.trim(y)

      y = y/np.max(np.abs(y))
      feature = extract_feature("live_audio.wav")
      feature = feature.reshape(1, -1)

      feature = scaler.transform(feature)

      prediction = model.predict(feature)
      emotion = label_encoder.inverse_transform(prediction)

      print("Predicted Emotion:", emotion[0])

predict_emotion_live()