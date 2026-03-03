from flask import Flask, render_template, request
import pickle
import librosa
import numpy as np
import os

app = Flask(__name__)

# Load trained files
model = pickle.load(open("svm_emotion_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

def extract_feature(file_path):
    audio, sample_rate = librosa.load(file_path, duration=6, offset=0.5)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    return np.hstack((mfcc_mean, mfcc_std))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["audio"]

    # Save inside static/uploads
    upload_folder = "static/uploads"
    os.makedirs(upload_folder, exist_ok=True)

    filepath = os.path.join(upload_folder, file.filename)
    file.save(filepath)

    # Feature extraction
    feature = extract_feature(filepath)
    feature = feature.reshape(1, -1)
    feature = scaler.transform(feature)

    prediction = model.predict(feature)
    emotion = label_encoder.inverse_transform(prediction)

    return render_template(
        "index.html",
        prediction_text="Predicted Emotion: " + emotion[0],
        audio_file=filepath   # 👈 send file path to HTML
    )

if __name__ == "__main__":
    app.run(debug=True)