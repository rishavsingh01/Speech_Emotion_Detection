from flask import Flask, render_template, request, jsonify
from flask import Response
import sounddevice as sd
import numpy as np
import cv2 

from speech_emotion import analyze_emotion,predict_emotion
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

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    try:
        audio, sr = record_audio(duration=9)

        timeline = analyze_emotion(audio, sr, predict_emotion)
        speech_emotion = timeline[-1]["emotion"]

        face_emotion = "Using Webcam"
        final = final_emotion(speech_emotion, face_emotion)

        result = {
            "speech_timeline": timeline,
            "face_emotion": face_emotion,
            "final_emotion": final
        }
        return jsonify(result)
    
    except Exception as e:
        print(f"Error processing emotions: {e}")
        return jsonify({"error": str(e)})
    
def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            #Detect emotion( already done in process route, so we can skip this step here)
            emotion = detect_face_emotion(frame)

            # putting emotion text on frame
            cv2.putText(frame,f"Emotion: {emotion}", (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # Encode frame 
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')            

if __name__ == '__main__':
    app.run(debug=True)
