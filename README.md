# Carla Lane Detection Project

> A real-time lane detection demo in the CARLA simulator, using classical CV methods enhanced with temporal smoothing and lane-area visualization.

---

## 🚀 Features

* 🎮 **CARLA Integration**: Runs in synchronous mode at a fixed timestep with Traffic Manager autopilot.
* 🧠 **Robust Lane Detection**:

  * **Adaptive ROI** separate trapezoid masks for left/right lanes tuned for road perspective.
  * **Slope filtering**: ignores near-horizontal and extreme-angle lines.
  * **Temporal smoothing**: exponential moving average to reduce jitter.
  * **Lane-area fill**: semi-transparent polygon between left and right lanes.
* 🖼️ **Debug Overlays**: live thumbnails of gray, edge, and masked images in the Pygame window.
* 📹 **Optional Recording**: Timestamped MP4 output of the main view.
* 🕹️ **Controls**:

  * **ESC** to quit.
  * **SPACE** to toggle autopilot on/off.
  * **W/S/A/D** for manual control when autopilot is disabled.

---

## 📸 Demo

![Lane Detection Demo](demo/lane_detection_demo.gif)

---

## 🛠️ Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/carla-lane-detection.git
cd carla-lane-detection

# (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🔧 Requirements

* Python 3.7
* CARLA Simulator (tested on v0.9.15)
* OpenCV (`opencv-python`)
* Pygame
* NumPy

> Note: CARLA must be running (Unreal engine) before launching the script.

---

## 🏃‍♂️ Usage

1. **Start CARLA** (in a separate terminal):

   ```bash
   ./CarlaUE4.sh  # or CarlaUE4.exe on Windows
   ```
2. **Run the demo**:

   ```bash
   python carla_lane_detection.py
   ```
3. **Options**:

   * Enable recording by passing `enable_recording=True` in the script or modifying the `__main__` call.

---

## 🔍 Code Structure

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

## 🧱 Algorithm Steps

1. **Preprocessing**:

   * Resize to 1080×720
   * Gaussian blur + Canny edge detection
2. **Region of Interest (ROI)**:

   * Two separate trapezoid masks to focus on left and right lanes perspective
3. **Line Detection**:

   * Hough transform to extract line segments
   * Filter by slope range (0.3–3.0) and side of image for left/right lanes
4. **Line Fitting & Smoothing**:

   * Average slope/intercept of segments
   * EMA smoothing across frames
5. **Visualization**:

   * Draw lane lines and fill lane area with semi-transparent polygon
   * Debug thumbnails for gray, edges, masked
6. **CARLA Integration**:

   * Synchronous stepping with `world.tick()`
   * Traffic Manager autopilot at specific FPS
   * Pygame display + input controls

---

## 🔄 Future Work

* Integrate **bird’s-eye-view** and sliding-window search for lane detection.
* Replace classical CV with **deep learning** (e.g., SCNN, YOLOv8-seg).
* Compare against CARLA’s **ground-truth lane topology** for quantitative evaluation.
* Add **curvature** and **vehicle offset** metrics on-screen.

---

## 📜 License

This project is licensed under the MIT License.

---

## 👤 Author

[Kostas Mantzaropoulos](https://github.com/kmatzaro)
