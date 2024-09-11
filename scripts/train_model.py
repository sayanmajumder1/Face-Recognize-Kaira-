import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
import pickle

def train_model(data_dir='dataset'):
    faces = []
    labels = []
    label_encoder = LabelEncoder()

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('jpg'):
                path = os.path.join(root, file)
                label = os.path.basename(root)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img_resized = cv2.resize(img, (150, 150))
                    faces.append(img_resized.flatten())
                    labels.append(label)
    
    if len(faces) == 0 or len(labels) == 0:
        print("No data found for training.")
        return

    if len(set(labels)) < 2:
        print("Error: The dataset must contain at least two classes.")
        return

    faces = np.array(faces)
    labels = label_encoder.fit_transform(labels)

    clf = svm.SVC(gamma='scale', probability=True)
    clf.fit(faces, labels)

    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/face_recognition_model.pkl'))
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump((clf, label_encoder), f)

    print("Model training complete and saved to:", model_path)

if __name__ == "__main__":
    train_model()
