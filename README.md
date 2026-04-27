# 🚀 Phase 3: Jetson Orin Nano Perception Pipeline (CSI Camera + TensorRT)

---

## 🎯 Objective

Migrate the perception pipeline from WSL to native hardware (Jetson Orin Nano), integrate CSI camera (IMX219), and optimize inference using TensorRT for real-time performance.

---

## 🏗️ What Was Implemented

* Setup Jetson Orin Nano with Ubuntu (JetPack)
* Integrated IMX219 CSI camera
* Rebuilt camera node using GStreamer (nvargus)
* Deployed YOLOv8 model on device
* Converted model → ONNX → TensorRT engine
* Built end-to-end pipeline: Camera → Detection → Display
* Integrated ROS2 launch system for full pipeline

---

### 🔹 Package Used

* `perception_pkg` → Contains camera, detector, display, and launch files

---

### 🔹 Nodes

#### 📷 Camera Node

* `camera_node` → Uses GStreamer pipeline for CSI camera

#### 🧠 Detector Node

* `detector_node` → Runs TensorRT engine for fast inference

#### 🧭 Tracker Node

* `tracker_node` → Tracks detected objects across frames (ID assignment + persistence)

#### 🖥️ Display Node

* `display_node` → Visualizes detections (bounding boxes + labels + track IDs)

---

### 🔹 Model Pipeline

* `yolov8n.pt` → Base model (PyTorch)
* `model.onnx` → Intermediate format
* `model.engine` → TensorRT optimized engine

---

### 🔹 Topics

* `/camera/image_raw`
* `/detections`
* `/processed_image`

---

## ⚙️ Environment Setup

* Device: Jetson Orin Nano
* OS: Ubuntu (JetPack SDK)
* JetPack Version: 5.x (L4T)
* ROS2: Humble
* Camera: IMX219 (CSI)

### Key Setup

```bash
source /opt/ros/humble/setup.bash
cd ~/perception_ws
colcon build
source install/setup.bash
```

---

## 🧪 Commands Used

```bash
# Launch full pipeline
ros2 launch perception_pkg perception_pipeline.launch.py

# Check camera
gst-launch-1.0 nvarguscamerasrc ! nveglglessink

# Debug
ros2 topic list
ros2 node list
```

---

## 🚧 Challenges Faced & Solutions

### 🔴 Issue 1: CSI Camera Not Working

**Problem:**
No output from camera

**Root Cause:**
Incorrect pipeline / camera not initialized

**Solution:**

```bash
gst-launch-1.0 nvarguscamerasrc ! nveglglessink
```

---

### 🔴 Issue 2: OpenCV Camera Access Fails

**Problem:**
`cv2.VideoCapture(0)` does not work

**Root Cause:**
CSI camera requires GStreamer, not direct access

**Solution:**
Used GStreamer pipeline inside OpenCV

---

### 🔴 Note: Model Format for Jetson

**Observation:**
Using `.pt` (PyTorch) directly led to suboptimal performance and compatibility issues on Jetson.

**Approach Used:**
Switched to TensorRT `.engine` file, which is optimized for Jetson GPU and provides better performance and compatibility.

**Final Setup:**

* `yolov8n.pt` → Converted to `model.engine`
* `model.engine` used directly in `detector_node`

---

### 🔴 Issue 7: PyTorch Installation Failed on Jetson

**Problem:**
Unable to install/import PyTorch; CUDA not detected or import errors while running:

```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```

**Observed Errors (examples):**

* Import errors from `torch._C`
* Missing CUDA / cuSPARSELt packages during install

**Root Cause:**

* Version mismatch between JetPack, CUDA, and PyTorch
* Jetson requires specific pre-built wheels (not standard pip install)

**Solution (Using Dusty-NV Containers):**
Used NVIDIA Jetson containers from Dusty-NV (jetson-containers) which provide pre-configured environments

```bash
# Install Docker (if not already)
sudo apt update
sudo apt install docker.io

# Clone Dusty-NV jetson containers repo
git clone https://github.com/dusty-nv/jetson-containers.git
cd jetson-containers

# Run PyTorch container
./run.sh dustynv/pytorch:latest
```

**Verification:**

```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```

---

### 🔴 Issue 8: TensorRT / CUDA Library Errors

**Problem:**
Errors like missing libraries during setup (e.g., `libcusparselt` not found) or runtime failures

**Root Cause:**
Incomplete/incorrect CUDA or JetPack components on host

**Solution:**

* Relied on Docker environment with pre-installed compatible libraries
* Avoided manual dependency resolution on host

---

## 📊 Results

* Achieved near real-time detection
* Significant FPS improvement using TensorRT
* Stable camera + detection pipeline on hardware

---

## 📊 Learnings

* Learned Jetson hardware ecosystem
* Understood GStreamer for CSI cameras
* Learned TensorRT optimization pipeline
* Improved debugging in embedded systems

---

## 🔮 Future Improvements

* Integrate depth estimation
* Optimize further using FP16 / INT8

---

## 📂 Folder Structure

```
perception_ws/
├── src/
│   └── perception_pkg
├── build/
├── install/
└── log/
```

---

## 💡 Key Takeaway

This phase transitioned the project from a simulated WSL environment to real embedded hardware, focusing on performance optimization and real-time inference.

---
