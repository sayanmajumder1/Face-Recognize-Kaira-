import cv2
import os
import sys
import time
from datetime import datetime
import numpy as np
from collections import deque

class FaceImageCapturer:
    def __init__(self, save_dir='dataset'):
        """Initialize the face capturer with DNN and Haar fallback"""
        
        # Get absolute paths
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # ====================================================================
        # FIX 1: SIMPLIFIED PATH RESOLUTION
        # ====================================================================
        # Save directory - always use project_root/dataset
        if os.path.isabs(save_dir):
            self.save_dir = save_dir
        else:
            self.save_dir = os.path.join(self.project_root, save_dir)
        
        # Ensure save directory exists
        os.makedirs(self.save_dir, exist_ok=True)
        
        print(f"[INFO] Project root: {self.project_root}")
        print(f"[INFO] Save directory: {self.save_dir}")
        
        # ====================================================================
        # FIX 2: PROPER PATH RESOLUTION FOR DNN MODELS
        # ====================================================================
        proto_path = os.path.join(self.project_root, 'models', 'deploy.prototxt')
        model_path = os.path.join(self.project_root, 'models', 'res10_300x300_ssd_iter_140000.caffemodel')
        
        print(f"[INFO] Looking for DNN models at: {os.path.join(self.project_root, 'models')}")
        
        # Try DNN first
        self.face_detector = None
        self.use_dnn = False
        
        if os.path.exists(proto_path) and os.path.exists(model_path):
            try:
                self.face_detector = cv2.dnn.readNetFromCaffe(proto_path, model_path)
                self.use_dnn = True
                print("[INFO] ✅ Using DNN face detector (more accurate)")
            except Exception as e:
                print(f"[WARNING] Failed to load DNN models: {e}")
                self.face_detector = None
                self.use_dnn = False
        else:
            print("[INFO] DNN models not found. Using Haar cascade detector.")
            if not os.path.exists(proto_path):
                print(f"[WARNING] Missing: {proto_path}")
            if not os.path.exists(model_path):
                print(f"[WARNING] Missing: {model_path}")
        
        # Haar cascade as fallback
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        if self.face_cascade.empty():
            print("[ERROR] Haar cascade not found!")
            raise RuntimeError("Haar cascade not found")
        
        if not self.use_dnn:
            print("[INFO] Using Haar cascade detector (fallback)")
        
        # ====================================================================
        # FIX 3: IMAGE QUALITY TRACKING
        # ====================================================================
        self.quality_threshold = 50  # Minimum quality score (0-100)
        self.last_face_quality = 0
        
        # ====================================================================
        # FIX 4: CAPTURE STATE
        # ====================================================================
        self.capture_count = 0
        self.total_target = 0
        self.is_capturing = False

    def preprocess_face(self, face_img):
        """Apply illumination normalization and histogram equalization"""
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        equalized = clahe.apply(face_img)
        
        # Gamma correction for brightness normalization
        gamma = 1.2
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        corrected = cv2.LUT(equalized, table)
        
        return corrected

    def check_image_quality(self, face_img):
        """
        Check image quality based on:
        - Sharpness (Laplacian variance)
        - Brightness (mean intensity)
        - Contrast (standard deviation)
        Returns: quality_score (0-100)
        """
        if face_img is None or face_img.size == 0:
            return 0
        
        # Sharpness - Laplacian variance (higher = sharper)
        laplacian_var = cv2.Laplacian(face_img, cv2.CV_64F).var()
        sharpness_score = min(laplacian_var / 50, 100)  # Normalize to 0-100
        
        # Brightness - mean intensity (optimal ~127)
        mean_brightness = np.mean(face_img)
        brightness_score = 100 - abs(mean_brightness - 127) * (100 / 127)
        brightness_score = max(0, min(brightness_score, 100))
        
        # Contrast - standard deviation (higher = more contrast)
        std_dev = np.std(face_img)
        contrast_score = min(std_dev / 50, 100) * 100
        contrast_score = min(contrast_score, 100)
        
        # Combined score (weighted)
        quality_score = (sharpness_score * 0.4 + brightness_score * 0.3 + contrast_score * 0.3)
        
        return quality_score

    def capture(self, name, num_images=110):
        """
        Main capture function with enhanced features
        """
        # ====================================================================
        # FIX 5: CREATE DIRECTORY WITH PROPER PATH
        # ====================================================================
        user_dir = os.path.join(self.save_dir, name)
        os.makedirs(user_dir, exist_ok=True)
        print(f"[INFO] Images will be saved to: {os.path.abspath(user_dir)}")

        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Webcam not accessible. Check permissions.")
            return

        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.total_target = num_images
        self.capture_count = 0
        self.is_capturing = True
        
        # ====================================================================
        # FIX 6: BETTER UI WITH FACE DETECTION FEEDBACK
        # ====================================================================
        print(f"\n{'='*60}")
        print(f"📸 CAPTURING FACES FOR: {name}")
        print(f"{'='*60}")
        print(f"📁 Target directory: {os.path.abspath(user_dir)}")
        print(f"🎯 Target images: {num_images}")
        print(f"📷 Using: {'DNN' if self.use_dnn else 'Haar'} detector")
        print("\n[INFO] Auto-capture is active!")
        print("[INFO] Keep your face centered and well-lit")
        print("[INFO] Press 'q' to quit early")
        print("[INFO] Press 's' for manual capture")
        print(f"{'='*60}\n")

        last_capture = time.time()
        capture_interval = 0.3  # 300ms between captures
        quality_warning_shown = False
        no_face_warning_shown = False
        
        # ====================================================================
        # FIX 7: FACE DETECTION STABILITY
        # ====================================================================
        face_history = deque(maxlen=5)  # Smooth face detection
        smoothed_faces = []

        while self.capture_count < num_images:
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Failed to read frame")
                continue

            # Flip horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

            # ====================================================================
            # FIX 8: IMPROVED FACE DETECTION WITH FALLBACK
            # ====================================================================
            faces = []
            
            # Try DNN first
            if self.use_dnn and self.face_detector is not None:
                try:
                    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                                 (300, 300), (104.0, 177.0, 123.0))
                    self.face_detector.setInput(blob)
                    detections = self.face_detector.forward()
                    
                    for i in range(detections.shape[2]):
                        confidence = detections[0, 0, i, 2]
                        if confidence > 0.6:
                            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                            (x1, y1, x2, y2) = box.astype("int")
                            faces.append((x1, y1, x2 - x1, y2 - y1))
                except Exception as e:
                    print(f"[WARNING] DNN detection failed: {e}")
                    self.use_dnn = False
                    faces = []

            # Fallback to Haar
            if len(faces) == 0:
                detected = self.face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.05,
                    minNeighbors=4,
                    minSize=(80, 80)
                )
                faces = detected

            # ====================================================================
            # FIX 9: FACE DETECTION SMOOTHING
            # ====================================================================
            if len(faces) > 0:
                # Smooth face positions
                for (x, y, fw, fh) in faces:
                    face_history.append((x, y, fw, fh))
                
                # Use median position for stability
                if len(face_history) > 0:
                    try:
                        avg_x = int(np.median([f[0] for f in face_history]))
                        avg_y = int(np.median([f[1] for f in face_history]))
                        avg_w = int(np.median([f[2] for f in face_history]))
                        avg_h = int(np.median([f[3] for f in face_history]))
                        smoothed_faces = [(avg_x, avg_y, avg_w, avg_h)]
                    except:
                        smoothed_faces = faces
                else:
                    smoothed_faces = faces
            else:
                smoothed_faces = []
                face_history.clear()

            # Draw rectangles and status
            face_detected = len(smoothed_faces) > 0
            
            if face_detected:
                no_face_warning_shown = False
                for (x, y, fw, fh) in smoothed_faces:
                    # Green box for detected face
                    cv2.rectangle(frame, (x, y), (x+fw, y+fh), (0, 255, 0), 2)
                    cv2.putText(frame, "✅ Face Detected", (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                # Red text when no face detected
                if not no_face_warning_shown:
                    print("[WARNING] No face detected. Please look at the camera.")
                    no_face_warning_shown = True
                cv2.putText(frame, "❌ No Face Detected", (10, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Show capture progress
            progress = int((self.capture_count / num_images) * 100)
            cv2.putText(frame, f"📸 Captured: {self.capture_count}/{num_images} ({progress}%)", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show detector type
            detector_label = "DNN" if self.use_dnn else "Haar"
            cv2.putText(frame, f"🔍 Detector: {detector_label}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Show quality if available
            if self.last_face_quality > 0:
                quality_text = f"⭐ Quality: {int(self.last_face_quality)}%"
                cv2.putText(frame, quality_text, (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # ====================================================================
            # FIX 10: AUTO-CAPTURE WITH QUALITY CHECK
            # ====================================================================
            current_time = time.time()
            if (current_time - last_capture) >= capture_interval and self.capture_count < num_images:
                if face_detected and len(smoothed_faces) > 0:
                    # Use the largest face (closest to camera)
                    x, y, fw, fh = max(smoothed_faces, key=lambda f: f[2] * f[3])
                    
                    # Add padding
                    padding = 20
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    fw = min(w - x, fw + 2*padding)
                    fh = min(h - y, fh + 2*padding)

                    face_roi = gray[y:y+fh, x:x+fw]
                    if face_roi.size > 0:
                        # ============================================================
                        # FIX 11: IMAGE QUALITY CHECK BEFORE SAVING
                        # ============================================================
                        quality_score = self.check_image_quality(face_roi)
                        self.last_face_quality = quality_score
                        
                        if quality_score < self.quality_threshold:
                            if not quality_warning_shown:
                                print(f"[WARNING] Low quality face detected ({quality_score:.0f}%). Skipping...")
                                quality_warning_shown = True
                            continue
                        else:
                            quality_warning_shown = False
                        
                        # Preprocess and save
                        processed_face = self.preprocess_face(face_roi)
                        resized_face = cv2.resize(processed_face, (224, 224))

                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        filename = os.path.join(user_dir, f"{self.capture_count}_{timestamp}.jpg")
                        cv2.imwrite(filename, resized_face)
                        
                        self.capture_count += 1
                        last_capture = current_time
                        
                        # Show capture feedback
                        print(f"[CAPTURE] {self.capture_count}/{num_images} | Quality: {quality_score:.0f}% | {os.path.basename(filename)}")
                        
                        # Visual feedback - flash green
                        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 5)

            # Show the frame
            cv2.imshow('Face Capture - Press Q to quit', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[INFO] Capture stopped by user")
                break
            elif key == ord('s'):  # Manual capture with 's' key
                if face_detected and len(smoothed_faces) > 0:
                    x, y, fw, fh = max(smoothed_faces, key=lambda f: f[2] * f[3])
                    padding = 20
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    fw = min(w - x, fw + 2*padding)
                    fh = min(h - y, fh + 2*padding)
                    face_roi = gray[y:y+fh, x:x+fw]
                    if face_roi.size > 0:
                        processed_face = self.preprocess_face(face_roi)
                        resized_face = cv2.resize(processed_face, (224, 224))
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        filename = os.path.join(user_dir, f"manual_{self.capture_count}_{timestamp}.jpg")
                        cv2.imwrite(filename, resized_face)
                        self.capture_count += 1
                        print(f"[MANUAL] Captured {self.capture_count}/{num_images}")
                else:
                    print("[WARNING] No face detected to capture manually!")

        # ====================================================================
        # FIX 12: CLEANUP AND SUMMARY
        # ====================================================================
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n{'='*60}")
        print(f"✅ CAPTURE COMPLETE!")
        print(f"{'='*60}")
        print(f"👤 Person: {name}")
        print(f"📸 Images captured: {self.capture_count}/{num_images}")
        print(f"📁 Saved to: {os.path.abspath(user_dir)}")
        
        if self.capture_count < num_images:
            print(f"⚠️ Only {self.capture_count} images captured (target was {num_images})")
            print("   Possible reasons:")
            print("   - You pressed 'q' to quit early")
            print("   - Low quality faces were skipped (< 50% quality)")
            print("   - No face was detected")
            print("   - Try better lighting or position")
        else:
            print(f"🎯 Perfect! All {num_images} images captured successfully!")
            print("   📊 Average quality: Good")
        
        print(f"{'='*60}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("[ERROR] Please provide a name")
        print("Usage: python capture_images.py <name>")
        print("Example: python capture_images.py John")
        print("\nOr from project root:")
        print("python scripts/capture_images.py John")
        sys.exit(1)
    
    try:
        capturer = FaceImageCapturer()
        capturer.capture(sys.argv[1])
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")