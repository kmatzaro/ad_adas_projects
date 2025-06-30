# Lane Detection in CARLA Simulator

> Real-time lane detection using classical computer vision techniques on synthetic driving scenes from the CARLA simulator. Features optional recording, debug overlays, and modular design for future upgrades like BEV transforms or deep learning.

---

## 🚀 Features

- 🎮 CARLA simulator integration with real-time lane detection
- 🧠 Classical CV: Gaussian blur, Canny edges, Hough lines, slope fitting
- 🧾 Optional video recording with timestamped filenames
- 🖼️ Debug visualizations (gray, edges, ROI mask) inside Pygame window
- 🧩 Modular design: easy to swap in AI models or BEV transforms later

---

## 📸 Demo

![Demo](demo/lane_detection_demo.gif)

---

## 🛠️ Installation

```bash
# Clone repo
git clone https://github.com/yourusername/carla-lane-detection.git
cd carla-lane-detection

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install Python dependencies
pip install -r requirements.txt
```

---

## 🔧 Requirements

- Python 3.8+
- CARLA Simulator (tested on 0.9.15)
- OpenCV
- Pygame
- NumPy

Automatically generated via:
```bash
pip freeze > requirements.txt
```

---

## 🧪 How to Run

Make sure CARLA is running in a separate terminal:
```bash
./CarlaUE4.sh  # or CarlaUE4.exe on Windows
```

Then start your app:
```bash
python carla_lane_detection.py
```

**Optional flags:**
- `enable_recording = True` in code → saves a timestamped `.mp4`

---

## 📂 Project Structure

```bash
├── carla_lane_detection.py       # Main app
├── camera_image_sensor.py        # This script captures frames from a drive storing them fo future processes
├── detection_yolo.py             # We perform inference on the image data from carla using YOLO
├── simple_lane_detection.py      # Classical CV lane detector
├── requirements.txt
├── demo/
│   └── lane_detection_demo.gif
└── README.md
```

---

## 🧱 Lane Detection Stages

1. Convert to grayscale
2. Apply Gaussian blur
3. Detect edges with Canny
4. Filter region of interest
5. Fit left/right lane lines using Hough transform
6. Overlay fitted lines on RGB frame

---

## 🔄 Future Plans

- [ ] Add bird’s eye view (BEV) and polynomial fits
- [ ] Use AI models like YOLO or SCNN for lane detection
- [ ] Validate against CARLA’s map/ground truth lanes
- [ ] Add lane curvature + driving logic

---

## 📜 License

MIT License

---

## 👤 Author

[Kostas Mantzaropoulos](https://github.com/kmatzaro)