import cv2
import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime

def recognize_face():
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/face_recognition_model.pkl'))
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return

    try:
        with open(model_path, 'rb') as f:
            clf, le = pickle.load(f)
    except (EOFError, pickle.UnpicklingError) as e:
        print(f"Failed to load model: {e}")
        return

    # Haar cascade face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)
    recognized_labels = load_logged_attendance()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (150,150)).flatten().reshape(1, -1)

            try:
                prediction = clf.predict(face_resized)
                label = le.inverse_transform(prediction)[0]
                color = (0, 255, 0)  # Green for known person
            except:
                label = "Unknown"
                color = (0, 0, 255)  # Red for unknown person

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            # Put the name of the recognized person above the rectangle
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            if label != "Unknown" and label not in recognized_labels:
                recognized_labels.add(label)
                log_recognition(label)

        cv2.imshow('Recognize Face', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def load_logged_attendance():
    now = datetime.now()
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../records'))
    log_file = os.path.join(log_dir, f'attendance_{now.strftime("%Y-%m-%d")}.xlsx')

    if os.path.exists(log_file):
        try:
            df = pd.read_excel(log_file, engine='openpyxl')
            return set(df['Name'].tolist())
        except Exception as e:
            print(f"Failed to load logged attendance: {e}")
            return set()
    else:
        return set()

def log_recognition(label):
    now = datetime.now()
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../records'))
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'attendance_{now.strftime("%Y-%m-%d")}.xlsx')

    data = {
        'Name': label,
        'Date': now.strftime("%Y-%m-%d"),
        'Time': now.strftime("%H:%M:%S"),
        'Day': now.strftime("%A")
    }

    new_entry = pd.DataFrame([data])

    try:
        if os.path.exists(log_file):
            df = pd.read_excel(log_file, engine='openpyxl')
            df = pd.concat([df, new_entry], ignore_index=True)
        else:
            df = new_entry

        df.to_excel(log_file, index=False, engine='openpyxl')
        print(f"Attendance logged in: {log_file}")
    except PermissionError as e:
        print(f"Permission denied: {e}")
    except Exception as e:
        print(f"Failed to log recognition: {e}")

if __name__ == "__main__":
    recognize_face()
