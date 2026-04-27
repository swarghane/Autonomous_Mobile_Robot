# 🤖 Autonomous Mobile Robot (Perception + Tracking + Voice Interaction)

---

## 🎯 Project Overview

This project focuses on building an autonomous mobile robot capable of perceiving its environment, detecting and tracking objects, and making movement decisions in real time.

The system evolves from foundational ROS2 learning to real-world deployment on Jetson Orin Nano, and is designed to further expand into human interaction using voice and conversational AI.

---

## 🧠 Key Features

* 📷 Real-time camera streaming (IMX219 - CSI)
* 🧠 Object detection using YOLOv8
* 🧭 Object tracking across frames
* ⚡ Optimized inference using TensorRT (`.engine`)
* 🚗 Movement decision-making (in progress)
* 🔌 Motor control via Arduino Uno R3 (in progress)
* 🤖 Deployment on Jetson Orin Nano

---

## 🏗️ System Architecture

Camera (Jetson) → Detection → Tracking → Decision Making → Arduino → Motors
      ↓
      TF + RViz2 Visualization

Future Layer:
Voice Input → AI Model → Action + Speech Output

---

## ⚙️ Tech Stack

### Core

* ROS2 (Humble)
* Python

### Perception

* OpenCV
* YOLOv8 (`yolov8n.pt`)
* TensorRT (`.engine`)

### Hardware

* Jetson Orin Nano
* IMX219 CSI Camera
* Arduino Uno R3
* 4WD / Ackermann chassis (in progress)
* LiDAR (planned)

### Communication

* Serial communication (Jetson ↔ Arduino)

---

## 📂 Project Structure

```bash
autonomous-robot/
├── my_robot/        # Current robot development workspace
├── README.md
```

---

## 🚀 How to Run (Jetson)

```bash
cd ~/my_robot
colcon build
source install/setup.bash

ros2 launch perception_pkg perception_pipeline.launch.py
```

---

## 🔗 Development Journey (Detailed Repos)

This project was built in structured phases:

### 📘 Phase 1: ROS2 Basics (WSL)

* Nodes, topics, services
* Environment setup and debugging
  👉 Repo: *(https://github.com/swarghane/ROS2_Phase1_Practice_WS)*

---

### 📘 Phase 2: Perception Pipeline (WSL)

* Camera, detection, display nodes
* TF + RViz2
* USB camera issues (usbipd, permissions)
  👉 Repo: *https://github.com/swarghane/ROS2_Phase2_Perception*

---

### 📘 Phase 3: Jetson Deployment

* IMX219 CSI camera integration
* TensorRT optimization (`.engine`)
* Real-time inference on Jetson
  👉 Repo: *https://github.com/swarghane/ROS2_Phase3_*

---

## 🚧 Current Challenges

* Integrating perception with motor control
* Real-time decision-making for movement
* Reliable tracking for follow-me behavior

---

## 📊 Current Status

* ✅ Perception pipeline complete (Jetson)
* 🔄 Motor control integration in progress
* 🔄 Robot chassis selection (4WD vs Ackermann)

---

## 🔮 Future Roadmap

### Phase 4: Mobility

* Motor control + movement logic
* Obstacle avoidance using LiDAR

### Phase 5: Human Interaction

* Voice input (mic)
* Speech output (speaker)
* Scene description

### Phase 6: Intelligence

* Follow-me mode
* Conversational AI integration

---

## 💡 Key Takeaway

This project demonstrates a structured journey from ROS2 fundamentals to building a real-world autonomous robot with perception, tracking, and decision-making capabilities.

---
