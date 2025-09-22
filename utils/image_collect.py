import os
import cv2
import uuid
import time

# Path to save collected images
IMAGE_PATH = r'D:\WORK\Python\Project\vsl_mediapipe\vsl_data\vsl_image'

# Labels for each class
labels = ['Toi', 'Ten La', 'Kien truc su', "Lap trinh vien "]

# Number of images to capture per label
number_of_images = 20

# Countdown time before starting capture for each label
countdown_time = 5

# Initialize camera
cap = cv2.VideoCapture(0)

for label in labels:
    img_path = os.path.join(IMAGE_PATH, label)
    os.makedirs(img_path, exist_ok=True)

    print(f"Collecting images for {label}")

    # Countdown once before capturing starts
    for t in range(countdown_time, 0, -1):
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.putText(frame, f"Get ready for '{label}' - Starting in {t}",
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        cv2.waitKey(1000)  # wait 1 second

    # Capture images continuously after countdown
    for imgnum in range(number_of_images):
        ret, frame = cap.read()
        if not ret:
            continue

        imagename = os.path.join(img_path, f"{label}.{str(uuid.uuid1())}.jpg")
        cv2.imwrite(imagename, frame)

        cv2.imshow('frame', frame)
        print(f"Captured image {imgnum+1}/{number_of_images} for {label}")

        # Short delay between captures so you can slightly adjust pose
        cv2.waitKey(500)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
