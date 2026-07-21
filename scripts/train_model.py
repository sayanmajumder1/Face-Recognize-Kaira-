import os
import cv2
import numpy as np
import pickle
from sklearn import svm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import random
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class FaceModelTrainer:
    def __init__(self, data_dir='dataset', model_dir='models'):
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = self._resolve_project_path(data_dir, 'dataset')
        self.model_dir = self._resolve_project_path(model_dir, 'models')
        os.makedirs(self.model_dir, exist_ok=True)
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95, whiten=True)
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.is_binary = False  # Flag for binary classification

    def _resolve_project_path(self, path, fallback_dir):
        """Resolve relative paths from the project root."""
        if os.path.isabs(path):
            return path

        candidate = os.path.join(self.project_root, path)
        if os.path.exists(candidate):
            return candidate

        fallback = os.path.join(self.project_root, fallback_dir)
        if os.path.exists(fallback):
            return fallback

        return os.path.abspath(path)

    def augment_data(self, image):
        """Apply moderate data augmentation to avoid overfitting"""
        augmented = []
        # Original
        augmented.append(image)
        # Horizontal flip
        augmented.append(cv2.flip(image, 1))
        # Small rotations (less aggressive)
        for angle in [-5, 5]:
            M = cv2.getRotationMatrix2D((image.shape[1]/2, image.shape[0]/2), angle, 1.0)
            rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
            augmented.append(rotated)
        # Brightness adjustments (subtle)
        for alpha in [0.9, 1.1]:
            adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
            augmented.append(adjusted)
        return augmented

    def load_and_preprocess_data(self):
        faces = []
        labels = []
        print("[INFO] Loading dataset...")

        # Walk through dataset directory
        for root, dirs, files in os.walk(self.data_dir):
            # Skip empty directories
            if not files:
                continue
                
            for file in files:
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    path = os.path.join(root, file)
                    label = os.path.basename(root)
                    
                    # Skip if label is empty or hidden directory
                    if not label or label.startswith('.'):
                        continue
                        
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        # Images on disk are already preprocessed with CLAHE + Gamma 1.2.
                        # We apply mild Gaussian blur to match recognition preprocessing,
                        # and resize to (150, 150) for LBPH.
                        blurred = cv2.GaussianBlur(img, (3, 3), 0)
                        img_resized = cv2.resize(blurred, (150, 150))
                        faces.append(img_resized)
                        labels.append(label)

        if len(faces) == 0:
            raise ValueError("[ERROR] No images found in dataset! Please capture some images first.")

        unique_labels = list(set(labels))
        print(f"[INFO] Found {len(faces)} images for {len(unique_labels)} people: {unique_labels}")
        
        # Check if we have multiple classes
        if len(unique_labels) < 2:
            print("[WARNING] Only 1 person found in dataset. Switching to BINARY classification mode.")
            print("[INFO] Model will recognize this person vs. Unknown faces.")
            self.is_binary = True
        else:
            self.is_binary = False
            print(f"[INFO] Multi-class mode: {len(unique_labels)} people to recognize.")

        return faces, labels

    def train(self):
        try:
            # Load data
            faces, labels = self.load_and_preprocess_data()

            # Augment and expand dataset (moderate augmentation)
            print("[INFO] Applying data augmentation...")
            augmented_faces = []
            augmented_labels = []
            
            for face, label in tqdm(zip(faces, labels), total=len(faces)):
                aug_faces = self.augment_data(face)
                augmented_faces.extend(aug_faces)
                augmented_labels.extend([label] * len(aug_faces))

            print(f"[INFO] Augmented dataset size: {len(augmented_faces)} images")

            # Encode labels
            numeric_labels = self.label_encoder.fit_transform(augmented_labels)
            print(f"[INFO] Encoded labels: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")

            # Train LBPH on augmented data
            print("[INFO] Training LBPH model...")
            self.recognizer.train(augmented_faces, np.array(numeric_labels))

            # Prepare features for SVM with advanced preprocessing
            print("[INFO] Extracting HOG features for SVM...")
            hog_features = []
            
            for face in tqdm(augmented_faces):
                # Resize to consistent size for HOG
                face_resized = cv2.resize(face, (128, 128))
                # Compute HOG features
                hog = cv2.HOGDescriptor((128, 128), (16, 16), (8, 8), (8, 8), 9)
                features = hog.compute(face_resized).flatten()
                hog_features.append(features)

            hog_features = np.array(hog_features)
            
            # Check for NaN or infinite values
            if np.isnan(hog_features).any() or np.isinf(hog_features).any():
                print("[WARNING] NaN or Inf values detected in HOG features. Replacing with zeros.")
                hog_features = np.nan_to_num(hog_features)

            # Split data for proper validation (no leakage)
            X_train, X_test, y_train, y_test = train_test_split(
                hog_features, numeric_labels, test_size=0.2, 
                random_state=42, stratify=numeric_labels if len(set(numeric_labels)) > 1 else None
            )
            
            print(f"[INFO] Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")

            # Normalize features on training set only
            print("[INFO] Normalizing features...")
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test) if len(X_test) > 0 else X_train_scaled

            # PCA reduction on training set only
            print("[INFO] Applying PCA dimensionality reduction...")
            X_train_pca = self.pca.fit_transform(X_train_scaled)
            X_test_pca = self.pca.transform(X_test_scaled) if len(X_test) > 0 else X_train_pca
            print(f"[INFO] Reduced features to {X_train_pca.shape[1]} dimensions")

            # Save the training PCA features per class for outlier distance checking
            train_features_by_class = {}
            for label_idx in range(len(self.label_encoder.classes_)):
                class_mask = (y_train == label_idx)
                train_features_by_class[label_idx] = X_train_pca[class_mask]

            # Handle special case for binary classification with only 1 class
            if self.is_binary and len(set(numeric_labels)) == 1:
                print("[INFO] Single class detected. Using One-Class SVM for anomaly detection.")
                from sklearn.svm import OneClassSVM
                
                # Use OneClassSVM for single class
                clf = OneClassSVM(
                    kernel='rbf',
                    gamma='scale',
                    nu=0.15  # Slightly more tolerant
                )
                clf.fit(X_train_pca)
                
                # Save models with special flag
                lbph_path = os.path.join(self.model_dir, 'lbph_model.yml')
                self.recognizer.write(lbph_path)

                svm_path = os.path.join(self.model_dir, 'face_recognition_model.pkl')
                with open(svm_path, 'wb') as f:
                    pickle.dump((clf, self.pca, self.scaler, self.label_encoder, True, train_features_by_class), f)

                print(f"[SUCCESS] Models saved to {self.model_dir}")
                print(f"  - LBPH: {lbph_path}")
                print(f"  - One-Class SVM: {svm_path}")
                print("[INFO] This model will recognize the known person and reject unknowns.")
                return

            # For multi-class or binary with 2+ classes
            # Train SVM with grid search-inspired parameters
            print("[INFO] Training SVM classifier...")
            
            # Adjust parameters based on dataset size
            if len(set(numeric_labels)) == 2:
                # Binary classification
                clf = svm.SVC(
                    kernel='rbf',
                    gamma='scale',
                    C=100,
                    class_weight='balanced',
                    probability=True,
                    random_state=42
                )
            else:
                # Multi-class classification
                clf = svm.SVC(
                    kernel='rbf',
                    gamma='scale',
                    C=10,
                    class_weight='balanced',
                    probability=True,
                    random_state=42,
                    decision_function_shape='ovo'
                )
            
            clf.fit(X_train_pca, y_train)

            # Evaluate if we have a test set
            if len(X_test) > 0 and len(set(y_test)) > 1:
                y_pred = clf.predict(X_test_pca)
                accuracy = accuracy_score(y_test, y_pred)
                print(f"[INFO] SVM Validation Accuracy: {accuracy:.2%}")
                
                # Show detailed classification report
                if len(set(y_test)) <= 10:
                    print("[INFO] Classification Report:")
                    print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
            else:
                print("[INFO] Skipping validation (not enough data or single class).")

            # Save all models
            lbph_path = os.path.join(self.model_dir, 'lbph_model.yml')
            self.recognizer.write(lbph_path)

            svm_path = os.path.join(self.model_dir, 'face_recognition_model.pkl')
            with open(svm_path, 'wb') as f:
                pickle.dump((clf, self.pca, self.scaler, self.label_encoder, False, train_features_by_class), f)

            print(f"\n[SUCCESS] Models saved to {self.model_dir}")
            print(f"  - LBPH: {lbph_path}")
            print(f"  - SVM + PCA: {svm_path}")
            print(f"[INFO] Training complete! Model is ready for use.")
            print(f"[INFO] Recognizable people: {list(self.label_encoder.classes_)}")
            
        except Exception as e:
            print(f"[ERROR] Training failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

if __name__ == "__main__":
    trainer = FaceModelTrainer()
    trainer.train()