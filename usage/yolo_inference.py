import cv2
from ultralytics import YOLO

# Load your trained model (or use yolov8n.pt for demo)
model = YOLO(r"D:\WORK\Python\Project\CV\vsl_word_reg\models\yolo\yolo_test_v3_s.pt")

# Open webcam (0 = default camera, or replace with video file path)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference on the frame
    results = model(frame)

    # Plot results on the frame
    annotated_frame = results[0].plot()

    # Show the frame
    cv2.imshow("YOLOv8 Real-Time Inference", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
