import sounddevice as sd
import numpy as np
import json

from speech_emotion import predict_emotion
from facial_emotion import detect_face_emotion


# 🎤 Record Audio
def record_audio(duration=3, fs=22050):
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    return audio.flatten(), fs


# 🔄 Fusion Logic
def final_emotion(speech_emotion, face_emotion):
    if speech_emotion == face_emotion:
        return speech_emotion
    else:
        return "Mixed Emotion"


# 🚀 Main Execution
if __name__ == "__main__":

    print("\n=== Emotion Recognition System ===\n")

    # Step 1: Record Audio
    audio, sr = record_audio(duration=6)

    # Step 2: Speech Emotion Timeline
    timeline = predict_emotion(audio, sr, predict_emotion)

    print("\nSpeech Emotion Timeline:")
    for t in timeline:
        print(t)

    # Step 3: Final Speech Emotion
    speech_emotion = timeline[-1]["emotion"]

    # Step 4: Face Emotion
    face_emotion = detect_face_emotion()

    print("\nFace Emotion:", face_emotion)

    # Step 5: Final Fusion
    final = final_emotion(speech_emotion, face_emotion)

    print("\nFinal Emotion:", final)

    # Save results
    result = {
        "speech_timeline": timeline,
        "face_emotion": face_emotion,
        "final_emotion": final
    }

    with open("outputs/results.json", "w") as f:
        json.dump(result, f, indent=4)

    print("\nResults saved in outputs/results.json")