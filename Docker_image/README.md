# AMR Docker Environment — NVIDIA Jetson Orin Nano

This directory contains the reproducible Docker environment used to run the **Autonomous Mobile Robot (AMR)** software stack on an **NVIDIA Jetson Orin Nano 8GB**.

The container provides the ROS 2, CUDA, computer-vision, speech, audio, and Python dependencies required by the robot while using the NVIDIA Jetson GPU for accelerated workloads such as TensorRT perception and Faster-Whisper/CTranslate2 speech recognition.

---

## Environment Overview

| Component | Configuration |
| --- | --- |
| Target Platform | NVIDIA Jetson Orin Nano 8GB |
| Architecture | ARM64 / AArch64 |
| Base Image | `dustynv/l4t-pytorch:r36.4.0` |
| Jetson Linux | L4T R36.x |
| ROS | ROS 2 Humble |
| Python | Python 3.10 |
| GPU Runtime | NVIDIA Container Runtime |
| Vision | OpenCV + Ultralytics + TensorRT |
| STT | Faster-Whisper |
| CTranslate2 | Custom GPU-enabled 4.8.1 AArch64 build |
| TTS | Piper |
| Audio | ALSA / PulseAudio / PortAudio |
| Camera | NVIDIA Argus / GStreamer |

The AMR source code itself is not baked into the image. The repository is mounted into `/workspace` when the container is started.

---

## Directory Structure

```text
Docker_image/
│
├── Dockerfile
├── apt_packages.txt
├── requirements.txt
│
├── scripts/
│   ├── build.sh
│   ├── run.sh
│   └── test.sh
│
├── wheels/
│   └── ctranslate2_gpu/
│       ├── README.md
│       ├── ctranslate2-4.8.1-cp310-cp310-linux_aarch64.whl
│       └── lib/
│           └── libctranslate2.so.4.8.1
│
└── README.md
```

---

# Dockerfile

The Docker image is based on:

```dockerfile
FROM dustynv/l4t-pytorch:r36.4.0
```

The Dockerfile configures:

- ROS 2 Humble
- ROS build tools
- OpenCV
- GStreamer
- Audio libraries
- Ultralytics
- Faster-Whisper
- NVIDIA NIM client dependencies
- ROS Python dependencies
- Custom GPU-enabled CTranslate2 4.8.1
- Additional AMR runtime dependencies

The base image already provides the NVIDIA Jetson CUDA/PyTorch environment required by the robot.

---

# Custom CTranslate2 GPU Build

Faster-Whisper uses **CTranslate2** as its inference runtime.

This project includes a GPU-enabled CTranslate2 build for:

```text
CTranslate2 : 4.8.1
Python      : CPython 3.10
Platform    : Linux AArch64
Target      : NVIDIA Jetson
```

The required files are stored under:

```text
wheels/ctranslate2_gpu/
```

During the Docker build, the shared library is installed into:

```text
/usr/local/lib/
```

and the Python wheel is installed using `pip`.

The Dockerfile then verifies that CTranslate2 can see CUDA using:

```python
import ctranslate2

print(ctranslate2.get_supported_compute_types("cuda"))
```

More information is available in:

```text
wheels/ctranslate2_gpu/README.md
```

---

# Host Requirements

Before building or running the container, the Jetson host should have:

- NVIDIA JetPack / Jetson Linux installed
- Docker
- NVIDIA Container Runtime
- ROS-compatible USB devices connected as required
- IMX219 CSI camera configured on the Jetson host
- RPLIDAR connected
- ESP32-C3 connected
- USB microphone/audio devices configured

Verify Docker:

```bash
docker --version
```

Verify the NVIDIA runtime:

```bash
docker info | grep -i runtime
```

---

# CSI Camera Setup

The **IMX219 CSI camera must first work on the Jetson host**.

If the CSI interface requires configuration, use NVIDIA Jetson-IO on the host:

```bash
sudo /opt/nvidia/jetson-io/jetson-io.py
```

After changing CSI configuration, reboot the Jetson if requested.

Restart the Argus camera service when required:

```bash
sudo systemctl restart nvargus-daemon
```

The current AMR camera configuration is:

```text
Sensor           : IMX219-120
Capture          : 1640 × 1232
Capture Rate     : 30 FPS
ROS Image Output : 640 × 480
```

A host-side GStreamer test can be performed with:

```bash
gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! \
'video/x-raw(memory:NVMM),width=1640,height=1232,format=NV12,framerate=30/1' ! \
nvvidconv ! xvimagesink
```

The container receives access to NVIDIA Argus through:

```text
/tmp/argus_socket
```

---

# Build the Docker Image

From the repository:

```bash
cd Docker_image
chmod +x scripts/build.sh
./scripts/build.sh
```

The build script creates:

```text
amr_base_image:latest
```

The build is intentionally performed with:

```text
--no-cache
```

to ensure that dependency installation and the custom CTranslate2 setup are rebuilt cleanly.

---

# Run the Container

From the **root of the AMR repository**:

```bash
chmod +x Docker_image/scripts/run.sh
./Docker_image/scripts/run.sh
```

The run script provides access to the Jetson GPU and robot hardware.

Important options include:

```text
--runtime nvidia
--network host
--ipc host
--privileged
--pid host
```

The script also exposes hardware interfaces including:

```text
/dev/video*
/dev/snd
/dev/bus/usb
/dev/i2c-*
/dev/ttyUSB0
/dev/ttyACM0
```

and mounts:

```text
/tmp/argus_socket
/tmp/.X11-unix
/run/udev
PulseAudio runtime files
AMR repository → /workspace
```

> The container is intentionally given broad hardware access because it directly interfaces with the CSI camera, LiDAR, ESP32, audio devices, display, and other robot hardware. This configuration is intended for the dedicated AMR development platform.

---

# Build the ROS 2 Workspace

Inside the running container:

```bash
cd /workspace

source /opt/ros/humble/setup.bash

rosdep update

colcon build --symlink-install

source install/setup.bash
```

---

# Test the Environment

The included test script verifies several critical components.

Inside the container:

```bash
cd /workspace/Docker_image

chmod +x scripts/test.sh

./scripts/test.sh
```

The test checks:

### PyTorch CUDA

```python
import torch
print(torch.cuda.is_available())
```

Expected:

```text
True
```

### CTranslate2 CUDA

```python
import ctranslate2
print(ctranslate2.get_supported_compute_types("cuda"))
```

The result should contain supported CUDA compute types.

### Faster-Whisper

```python
from faster_whisper import WhisperModel
print("Whisper OK")
```

### ROS 2

```bash
ros2 doctor
```

---

# Verify GPU-Accelerated Whisper

The current STT node uses:

```text
Model        : tiny.en
Device       : cuda
Compute Type : int8_float16
```

A simple standalone test is:

```bash
python3 - <<'PY'
from faster_whisper import WhisperModel

model = WhisperModel(
    "tiny.en",
    device="cuda",
    compute_type="int8_float16"
)

print("Faster-Whisper CUDA model loaded successfully")
PY
```

---

# Robot Models

Large runtime models are intentionally kept outside Git.

The robot expects models under the repository-local:

```text
models/
```

Typical structure:

```text
models/
├── vision/
│   └── yolov8n-pose.engine
│
└── piper/
    ├── en_US-hfc_female-medium.onnx
    ├── en_US-hfc_female-medium.onnx.json
    ├── en_US-hfc_male-medium.onnx
    └── en_US-hfc_male-medium.onnx.json
```

The ROS workspace is mounted at `/workspace`, so these become available inside the container as:

```text
/workspace/models/...
```

---

# Piper Voice Models

The current TTS implementation supports:

```text
en_US-hfc_female-medium
en_US-hfc_male-medium
```

Piper voice models can be downloaded into the local model directory using:

```bash
python3 -m piper.download_voices \
    --data-dir /workspace/models/piper \
    en_US-hfc_female-medium
```

or:

```bash
python3 -m piper.download_voices \
    --data-dir /workspace/models/piper \
    en_US-hfc_male-medium
```

---

# Environment Variables

The NVIDIA API key used by the interaction system is loaded from:

```text
/workspace/.env
```

Create a local `.env` file in the root of the repository:

```text
NVIDIA_API_KEY=your_nvidia_api_key_here
```

Never commit real API keys, credentials, tokens, or passwords to Git.

---

# Hardware Interfaces

Typical interfaces used by the AMR are:

| Hardware | Interface |
| --- | --- |
| IMX219 CSI camera | NVIDIA Argus / CSI |
| RPLIDAR C1 | USB serial |
| ESP32-C3 | `/dev/ttyACM0` |
| USB microphone | `/dev/snd` |
| Audio output | ALSA / PulseAudio |
| GPU | NVIDIA Container Runtime |

Actual Linux device names can vary between systems.

---

# Troubleshooting

## CTranslate2 does not detect CUDA

Run:

```bash
python3 -c "import ctranslate2; print(ctranslate2.get_supported_compute_types('cuda'))"
```

Also verify:

```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```

If PyTorch reports CUDA but CTranslate2 does not, verify that the custom wheel and shared library from:

```text
wheels/ctranslate2_gpu/
```

were installed during the Docker build.

---

## CSI camera does not open

First test the camera on the **Jetson host**, not inside the container.

Restart Argus:

```bash
sudo systemctl restart nvargus-daemon
```

Then verify:

```bash
gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! \
'video/x-raw(memory:NVMM),width=1640,height=1232,format=NV12,framerate=30/1' ! \
nvvidconv ! xvimagesink
```

Also confirm that the container has access to:

```text
/tmp/argus_socket
```

---

## Serial device not found

Check the Jetson host:

```bash
ls /dev/ttyACM*
ls /dev/ttyUSB*
```

Then confirm the corresponding device is exposed to the container.

---

## Audio is unavailable

Check host audio devices:

```bash
aplay -l
arecord -l
```

Inside the container:

```bash
aplay -l
arecord -l
```

The run script mounts the required sound device and PulseAudio runtime paths.

---

# Purpose of This Image

This Docker environment is designed specifically to make the AMR software stack reproducible on the NVIDIA Jetson platform.

It provides a single environment for:

```text
ROS 2
  +
TensorRT Vision
  +
YOLOv8 Pose
  +
LiDAR Processing
  +
Faster-Whisper
  +
CTranslate2 CUDA
  +
NVIDIA NIM Interaction
  +
Piper TTS
  +
ESP32 Communication
```

The result is a containerized runtime for the complete perception, decision, interaction, safety, visualization, and hardware-control stack of the robot.


## License

Original code in this project is licensed under the [MIT License](LICENSE).

Third-party libraries, ROS packages, models, and redistributed binary components remain subject to their respective upstream licenses.