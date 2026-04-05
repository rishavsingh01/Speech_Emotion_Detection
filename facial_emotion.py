from deepface import DeepFace
import cv2

def detect_face_emotion():
    cap = cv2.VideoCapture(0)

    ret, frame = cap.read()

    if ret:
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']
        except:
            emotion = "No face detected"
    else:
        emotion = "Camera Error"

    cap.release()
    return emotion