import os
from ultralytics import YOLO

def main():
    # Paths
    yaml_file = r"D:\WORK\Python\Project\vsl_mediapipe\vsl_data\yolo_labeled\data.yaml"
    val_path = r"D:\WORK\Python\Project\vsl_mediapipe\vsl_data\yolo_labeled\images\val"

    # === 1. Train YOLO on your dataset ===
    model = YOLO("yolov8l.pt")   # start from pretrained COCO weights

    model.train(
        data=yaml_file,           # your existing dataset config
        epochs=70,
        imgsz=640,
        batch=16,
        name="vsl_sign_detector",
        workers=2,
        device=0
    )

    # === 2. Validate trained model ===
    # Load the best weights from training
    trained_model = YOLO("runs/detect/vsl_sign_detector/weights/best.pt")
    metrics = trained_model.val(data=yaml_file)
    print("Validation results:", metrics)

    # === 3. Test on a val image ===
    test_img = os.path.join(val_path, os.listdir(val_path)[0])
    results = trained_model.predict(test_img, save=True, conf=0.5, device=0)
    print("✅ Prediction done, check 'runs/detect/predict'")

if __name__ == "__main__":
    main()
