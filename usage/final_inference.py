import cv2
from ultralytics import YOLO

# Path to your trained model (best.pt from training run)
model_path = r"D:\WORK\Python\Project\vsl_mediapipe\train\runs\detect\vsl_sign_detector\weights\best.pt"

# Load your trained YOLO model
model = YOLO(model_path)

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference
    results = model(frame, conf=0.25)  

    # Plot results on the frame
    annotated_frame = results[0].plot()

    cv2.imshow("YOLOv8 Realtime", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
