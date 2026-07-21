import os
import urllib.request
import ssl
import sys

def download_with_retry(url, filename, max_retries=3):
    """Download with retry and SSL context"""
    ssl._create_default_https_context = ssl._create_unverified_context
    
    for attempt in range(max_retries):
        try:
            print(f"[INFO] Attempt {attempt+1}/{max_retries} downloading {filename}...")
            urllib.request.urlretrieve(url, filename)
            print(f"[SUCCESS] Downloaded {filename}")
            return True
        except Exception as e:
            print(f"[WARNING] Attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                print("[INFO] Retrying in 2 seconds...")
                import time
                time.sleep(2)
    return False

def manual_download_dnn():
    """Download DNN models with fallback methods"""
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    # Option 1: Direct download (may fail due to DNS)
    files = {
        'deploy.prototxt': 'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt',
        'res10_300x300_ssd_iter_140000.caffemodel': 'https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel'
    }
    
    print("\n" + "="*60)
    print("MANUAL DNN MODEL DOWNLOADER")
    print("="*60)
    print(f"[INFO] Models will be saved to: {model_dir}\n")
    
    # Option 1: Try direct download
    success = True
    for filename, url in files.items():
        filepath = os.path.join(model_dir, filename)
        if os.path.exists(filepath):
            print(f"[INFO] {filename} already exists - skipping")
            continue
        
        print(f"[INFO] Downloading {filename}...")
        if not download_with_retry(url, filepath):
            success = False
            print(f"[ERROR] Failed to download {filename}")
    
    if success:
        print("\n[SUCCESS] All DNN models downloaded!")
        print("[INFO] You can now run recognize_face.py with DNN detector.")
        return True
    
    # Option 2: If direct download fails, provide manual instructions
    print("\n" + "="*60)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("="*60)
    print("\nSince automatic download failed, please manually download:")
    print("\n1. deploy.prototxt")
    print("   URL: https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt")
    print("   Save to: " + os.path.join(model_dir, 'deploy.prototxt'))
    print("\n2. res10_300x300_ssd_iter_140000.caffemodel")
    print("   URL: https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel")
    print("   Save to: " + os.path.join(model_dir, 'res10_300x300_ssd_iter_140000.caffemodel'))
    print("\n[INFO] Alternative: Use a browser to download these files")
    print("[INFO] Once downloaded, place them in the 'models' folder")
    print("\n[INFO] After manual download, restart recognize_face.py")
    
    return False

if __name__ == "__main__":
    manual_download_dnn()
    input("\nPress Enter to exit...")