# Carla Lane Detection Project

> A configurable, real-time lane detection & validation pipeline in the CARLA simulator, driven by YAML, with live Pygame visualization, robust classical-CV detection, and automatic ground-truth metrics logging.

---

## 🚀 Key Features

- **YAML Configuration**  
  All parameters for CARLA setup, camera, lane detector, and validation live in a single `config.yaml`. Tweak FOV, resolutions, thresholds, sync mode, capture intervals, and more without touching code.

- **Robust Classical-CV Pipeline**  
  - Gaussian blur → Canny edges → Hough lines  
  - Slope filtering (ignore near-horizontal/extreme lines)  
  - Exponential Moving Average smoothing + fallback for missing frames  
  - Semi-transparent lane-area fill and center-line overlay  

- **Automated Validation & Metrics**  
  - Samples CARLA waypoints, projects to 2D, interpolates predicted midpoints  
  - Computes per-frame mean error, RMSE, percentage within pixel threshold  
  - Saves validation CSV and per-frame overlay PNGs  

- **Live Debug Overlays**  
  - In-window thumbnails of gray, edge, and masked ROI images  
  - Adjustable via config flags  

- **Optional Video Recording**  
  - Timestamped MP4 output of the main view  
  - Controlled via a simple boolean in `config.yaml`

- **CARLA Integration & Controls**  
  - Synchronous mode at fixed delta (e.g. 20 fps) with Traffic Manager autopilot  
  - Keyboard:  
    - **ESC**: quit  
    - **SPACE**: toggle autopilot  
    - **W/S/A/D**: manual throttle/brake/steer when autopilot is off  

- **Modular Codebase**  
  - `SimpleLaneDetector` for detection  
  - `LaneValidator` for ground-truth comparison  
  - `CarlaLaneDetection` for simulation loop & display  

---

## 📸 Demo

![Lane Detection Demo](demo/lane_detection_demo.gif)

---

## 🛠️ Installation

```bash
git clone https://github.com/kmatzaro/perception_autonomous_driving.git
cd perception_autonomous_driving

# (Optional) Create & activate a virtual environment
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> **Note:** Make sure the CARLA server (Unreal Engine) is running before launching.

---

## ⚙️ Configuration

All tunable settings live in **`config.yaml`** at the project root. Example structure:

```yaml
carla:
  host: "localhost"
  port: 2000
  timeout: 10.0
  town: "Town03"
  validation_mode: False
  enable_recording: False
  FPS: 30
  pygame_display: # It better match image_resize resolution
    display_width: 1280 
    display_height: 720
  camera:
    image_width: 1920
    image_height: 1080
    fov: 90
    transform:
      location: { x: 1.5, y: 0.0, z: 1.3 }
      rotation: { pitch: -8, yaw: 0.0, roll: 0.0 }

lane_detector:
  image_resize:
    image_width: 1280
    image_height: 720
  gaussian_blur:
    kernel_size_x: 3
    kernel_size_y: 3
    sigma_x: 5
  canny:
    low_thresh: 50
    high_thresh: 150
  hough: 
    rho: 1
    threshold: 30
    min_line_len: 40
    max_line_gap: 100
  smoothing_factor: 0.9
  max_missing: 20
  display_lane_overlay: True
  display_lane_lines : True
  display_center_lane_line : True

validation:
  output_dir: "validation"
  threshold_px: 10
  num_captures: 30
  interval_seconds: 5.0
  y_min_pct: 0.6
  log_csv: "metrics.csv"
  draw_det_vs_gt: True
```

---

## 🚀 Usage

1. **Launch CARLA** in a separate terminal:  
   ```bash
   ./CarlaUE4.sh      # Linux/macOS
   CarlaUE4.exe       # Windows
   ```
2. **Run the app**:  
   ```bash
   python carla_lane_detection.py
   ```
3. **Controls**:  
   - **ESC**: quit  
   - **SPACE**: toggle autopilot  
   - **W/S/A/D**: manual control when autopilot is off  

---

## 📂 Project Structure

```bash
├── carla_lane_detection.py       # Main simulation & display loop
├── camera_image_sensor.py        # This script captures frames from a drive storing them fo future processes
├── detection_yolo.py             # We perform inference on the image data from carla using YOLO
├── simple_lane_detection.py      # Classical CV lane detector
├── validation_lane_detection.py  # Ground-truth validation & metrics
├── requirements.txt
├── demo/
│   └── lane_detection_demo.gif
└── README.md              
```

---

## 📊 Validation & Metrics

- **Mean error** (pixels)  
- **RMSE** (pixels)  
- **Percentage within threshold** (e.g. 10 px)  

Results are saved to `validation/metrics.csv` and overlays under `validation/frame_XXXX.png`.

---

## 🔄 Future Work

- Add **bird’s-eye-view** & sliding-window detection  
- Swap in **deep-learning** (e.g. SCNN, YOLOv8-seg)  
- Compute **lane curvature** & **vehicle offset** in real time  
- Integrate into **CI pipelines** for automated regression testing  

---

## 📜 License

MIT License

---

## 👤 Author

[Kostas Mantzaropoulos](https://github.com/kmatzaro)
