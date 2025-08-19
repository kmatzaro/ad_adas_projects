# CARLA Advanced Perception System

> A high-performance, real-time autonomous vehicle perception system combining classical computer vision and modern deep learning in CARLA simulator. Features dual-pipeline lane detection + object detection with GPU acceleration, BEV transformation, and comprehensive validation metrics.

---

## Demo
![Lane Detection Demo](perception_ad/demo/perception_demo.gif)

## 🚀 Key Features

### **Advanced Perception Pipeline**
- **Classical Lane Detection:** Gaussian blur → Canny edges → Hough lines with temporal smoothing
- **Modern Object Detection:** YOLO11 GPU-accelerated detection for vehicles, pedestrians, traffic signs  
- **Bird's Eye View (BEV) Transformation:** Optional perspective transformation for enhanced lane analysis
- **Real-time Performance:** <25ms total processing on newer GPUs

### **YAML-Driven Configuration**  
All parameters live in a single `config.yaml` - camera settings, detection thresholds, BEV parameters, validation options. Modify detection behavior without touching code.

### **Comprehensive Validation & Metrics**  
- **Automated Ground Truth:** Extracts CARLA waypoints, projects to 2D, measures pixel accuracy
- **Quantitative Metrics:** Per-frame mean error, RMSE, detection success rates
- **Visual Validation:** Overlay PNGs showing detected vs ground truth with metrics
- **Performance Monitoring:** Component-level timing breakdown, real-time compliance tracking

### **Professional Development Features**
- **Live Debug Overlays:** Real-time thumbnails of processing pipeline stages
- **Video Recording:** Timestamped MP4 output with detection overlays
- **Error Recovery:** Robust connection management, camera restart capabilities
- **Performance Analytics:** Detailed timing metrics for optimization

### **CARLA Integration & Controls**  
- **Synchronous Mode:** Fixed timestep simulation for reproducible results
- **Traffic Generation:** Configurable vehicle spawning for object detection testing
- **Interactive Controls:**
  - **ESC:** Quit application  
  - **SPACE:** Toggle autopilot on/off
  - **W/A/S/D:** Manual vehicle control

---

## 🏎️ Performance

### **Real-time Capabilities**
- **Lane Detection:** ~10ms (classical CV pipeline)
- **Object Detection:** ~20ms (YOLO11s on GTX 1660 Super GPU)
- **Total Perception:** <30ms (under 33ms real-time budget for target 30 FPS)
- **System FPS:** 30+ FPS sustained with full perception + visualization

### **Detection Capabilities**
- **Lane Lines:** Robust detection with temporal smoothing, missing frame handling
- **Objects Detected:** Cars, pedestrians, bicycles, motorcycles, buses, trucks, traffic lights, stop signs
- **BEV Support:** Optional bird's eye view transformation for improved lane geometry

### **System Requirements (my setup)**
- **GPU:** NVIDIA GPU with CUDA support (GTX 1660+ with 6GB VRAM)
- **Python:** 3.10 (CARLA 0.9.15 compatibility)
- **CARLA:** 0.9.15 required
- **Memory:** 32GB RAM

---

## 🛠️ Installation

```bash
git clone https://github.com/your-username/carla-perception-system.git
cd carla-perception-system

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install --no-deps ultralytics 
pip install -r requirements.txt

# Install CARLA (if needed)
pip install carla==0.9.15
```

> **Important:** Ensure CARLA server is running before launching the perception system.

---

## ⚙️ Configuration

All settings are configured through `config.yaml`:

### **Core System Settings**
```yaml
carla:
  host: "localhost"
  port: 2000
  town: "Town03"
  FPS: 30
  validation_mode: false
  enable_recording: false
  enable_debugs: true

  camera:
    image_width: 960
    image_height: 540
    fov: 90
    transform:
      location: { x: 1.5, y: 0.0, z: 1.3 }
      rotation: { pitch: -8, yaw: 0.0, roll: 0.0 }
```

### **Object Detection Configuration**
```yaml
object_detection:
  model_size: '11s'           # YOLO11: n/s/m/l/x (speed vs accuracy trade-off)
  confidence_threshold: 0.7   # Detection confidence (0.5-0.9)
  nms_threshold: 0.5         # Non-maximum suppression
```

### **BEV Transformation (Optional)**
```yaml
bev_lane_detector:
  bev_enabled: true          # Enable bird's eye view transformation
  bev_size:
    bev_width: 256
    bev_height: 256
  src_pts_frac:             # Source region as image fractions [0,1]
    - [0.0, 1.0]            # bottom-left
    - [1.0, 1.0]            # bottom-right  
    - [1, 0.45]             # right horizon
    - [0, 0.45]             # left horizon
```

### **Validation Pipeline**
```yaml
validation:
  output_dir: "validation"
  threshold_px: 10           # Accuracy threshold in pixels
  num_captures: 20           # Number of validation frames
  interval_seconds: 10.0     # Time between captures
  draw_det_vs_gt: true      # Generate overlay visualizations
```

---

## 🚀 Usage

### **1. Start CARLA Server**
```bash
# Linux/macOS
./CarlaUE4.sh

# Windows  
CarlaUE4.exe

# Alternative: Use CARLA package
carla-simulator
```

### **2. Run Perception System**
```bash
python carla_perception_main.py
```

### **3. Controls**
- **ESC:** Exit application
- **SPACE:** Toggle autopilot mode
- **W/A/S/D:** Manual vehicle control (when autopilot disabled)

### **4. Monitor Performance**
Real-time performance metrics are logged every 60 frames:
```
Enhanced Perception Performance (60 frames):
  Lane Detection    : avg=10.2ms  (8.1-15.7ms)
  Object Detection  : avg=13.8ms  (11.2-18.4ms) 
  Total Perception  : avg=24.0ms  (19.8-32.1ms)
  Callback Overhead : avg=2.1ms   (1.5-4.2ms)
  Real-time Status  : GOOD (target: <33.3ms)
```

---

## 📂 Project Structure

```
perception-ad/
├── carla_perception_main.py      # Main CARLA application and coordination
├── enhanced_perception.py        # Dual perception system coordinator  
├── simple_lane_detection.py      # Classical CV lane detection pipeline
├── object_detection.py           # YOLO-based object detection
├── bev_transformer.py            # Bird's eye view transformation
├── validation_lane_detection.py  # Ground truth validation pipeline
├── config.yaml                   # System configuration
├── requirements.txt              # Python dependencies
├── recordings/                   # Video output directory (created at runtime)
├── demo/
|   ├── generate_gif.py           # Python script that makes a gif out of a video
|   └── perception_demo.gif       $ Demo perception GIF
├── validation_*/                 # Validation results (timestamped)
|   ├── overlays/                 # Detection vs ground truth visualizations
|   └── logs/                     # Quantitative metrics and reports
└── README.md
```

---

## 📊 Validation & Metrics

### **Automated Validation Pipeline**
- **Ground Truth Extraction:** Samples CARLA HD map waypoints along vehicle trajectory
- **Camera Projection:** Projects 3D waypoints to 2D image coordinates using camera intrinsics
- **Accuracy Measurement:** Computes pixel-level errors between detected and ground truth centerlines
- **Statistical Analysis:** Mean error, RMSE, accuracy percentages, detection success rates

### **Output Metrics**
```csv
frame_id,timestamp,mean_error_px,rmse_px,pct_within_10px,detection_success
0,15.2,8.3,12.1,85.7,true
1,25.4,7.9,11.8,87.2,true
```

### **Visual Validation**
- **Overlay Images:** Ground truth (yellow) vs detected (red) centerlines
- **Metrics Annotation:** Real-time accuracy statistics on each frame
- **Error Visualization:** Clear identification of detection vs reality discrepancies

---

## 🔧 Troubleshooting

### **GPU/CUDA Issues**
```python
# Verify CUDA setup
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

**Solutions:**
- Install CUDA-enabled PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu121`
- Check GPU memory with `nvcc --version`
- Reduce model size: Change `model_size: '11n'` for lower memory usage

### **Performance Issues**
**Symptoms:** >33ms processing time, frame drops, stuttering

**Solutions:**
- **Reduce resolution:** Lower `image_width/height` in config
- **Lighter model:** Use `model_size: '11n'` instead of `'11s'`
- **Disable features:** Set `bev_enabled: false`, `enable_debugs: false`

### **Detection Quality Issues**
**Poor distant object detection:**
- Increase `image_width/height` for better resolution
- Decrease `confidence_threshold` to 0.5-0.6
- Use larger model: `model_size: '11m'`

**False positives:**
- Increase `confidence_threshold` to 0.8+
- Adjust `nms_threshold` for better duplicate filtering

### **CARLA Connection Issues**
**Server connection failures:**
- Verify CARLA server is running: `netstat -an | grep 2000`
- Check `host/port` in config.yaml
- Try different CARLA town: `town: "Town01"`

---

## 🎯 Advanced Features

### **BEV Transformation Calibration**
Adjust `src_pts_frac` for different camera angles:
- **High-mounted camera:** Increase horizon y-values (0.6-0.7)
- **Low-mounted camera:** Decrease horizon y-values (0.3-0.4)  
- **Wide FOV:** Adjust corner x-values for perspective correction

### **Multi-Weather Testing**
```python
# Add weather control to CARLA setup
weather = carla.WeatherParameters(
    cloudiness=80.0,
    precipitation=30.0,
    fog_density=10.0
)
world.set_weather(weather)
```

### **Custom Object Classes**
Modify `detection_classes` in `object_detection.py` for specific use cases:
```python
detection_classes = {
    0: 'person',
    2: 'car', 
    9: 'traffic_light',
    # Add custom classes as needed
}
```

---

## 🔄 Development Roadmap

### **Current Status ✅**
- ✅ Real-time dual perception (lanes + objects)
- ✅ BEV transformation support
- ✅ GPU-accelerated YOLO detection  
- ✅ Comprehensive validation pipeline
- ✅ Professional performance monitoring

### **Planned Enhancements**
- **Advanced Features:**
  - [ ] Object tracking between frames
  - [ ] Distance estimation for detected objects
  - [ ] Lane curvature and vehicle offset calculation
  - [ ] Weather condition adaptation

- **System Improvements:**
  - [ ] Multi-camera sensor fusion
  - [ ] Integration with vehicle control systems
  - [ ] Real-time parameter tuning interface
  - [ ] Advanced weather condition testing

---

## 📜 License

MIT License - See LICENSE file for details

---

## 👤 Author

**Kostas Mantzaropoulos** 
email: kmatzaro@gmail.com

---
