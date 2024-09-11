import cv2
import os
import sys

def capture_images(name, num_images=110, save_dir='dataset'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    user_dir = os.path.join(save_dir, name)
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)

    cap = cv2.VideoCapture(0)
    count = 0
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            continue
        resized_frame = cv2.resize(frame, (150, 150))
        cv2.imshow('Capture Images', resized_frame)
        cv2.imwrite(f'{user_dir}/{count}.jpg', resized_frame)
        count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Image capture complete. Images stored for training.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python capture_images.py <name>")
        sys.exit(1)
    name = sys.argv[1]
    capture_images(name)
