from ultralytics import YOLO
from pathlib import Path
import csv
import carla.PythonAPI.my_projects.perception_autonomous_driving.camera_image_sensor as camera_image_sensor

# Load YOLO model
model = YOLO("yolov8l.pt")

# Define paths
image_dir = Path("image_data")
output_dir = Path("detected_data")
log_file = output_dir / "detection_log.csv"

camera_image_sensor.cleanup_output_directory(output_dir)

# Create output directory
output_dir.mkdir(parents=True, exist_ok=True)

# Prepare CSV log
with open(log_file, mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Filename", "Num Detections", "Classes", "Confidences"])

    # Loop through all images
    for image_path in sorted(image_dir.glob("*.png")):
        print(f"Processing: {image_path.name}")
        result = model(str(image_path), save=True, project=str(output_dir), name="inference", exist_ok=True)[0]

        # Extract detection info
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes else []
        confs = result.boxes.conf.cpu().numpy() if result.boxes else []
        classes = result.boxes.cls.cpu().numpy() if result.boxes else []

        # Log to CSV
        writer.writerow([
            image_path.name,
            len(boxes),
            [int(c) for c in classes],
            [round(float(c), 3) for c in confs]
        ])