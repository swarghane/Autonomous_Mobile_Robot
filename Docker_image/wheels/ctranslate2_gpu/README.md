# GPU-Enabled CTranslate2 4.8.1 — Jetson AArch64

This directory contains the **CTranslate2 4.8.1 artifacts used by the AMR Docker environment for GPU-accelerated Faster-Whisper inference on NVIDIA Jetson**.

The files target:

```text
Platform     : Linux
Architecture : AArch64 / ARM64
Python       : CPython 3.10
CTranslate2  : 4.8.1
Target       : NVIDIA Jetson
Backend      : CUDA
```

---

## Purpose

The AMR uses **Faster-Whisper** for local Speech-to-Text.

Faster-Whisper uses **CTranslate2** as its inference engine.

The robot's current STT configuration is:

```text
Whisper Model : tiny.en
Device        : CUDA
Compute Type  : int8_float16
```

These CTranslate2 artifacts are included with the project to provide the GPU-enabled AArch64 runtime used by the tested Jetson environment.

---

# Directory Contents

```text
ctranslate2_gpu/
│
├── README.md
│
├── ctranslate2-4.8.1-cp310-cp310-linux_aarch64.whl
│
└── lib/
    └── libctranslate2.so.4.8.1
```

### Python Wheel

```text
ctranslate2-4.8.1-cp310-cp310-linux_aarch64.whl
```

Wheel tags:

```text
Version      : 4.8.1
Python       : CPython 3.10
Architecture : Linux AArch64
```

### Shared Library

```text
lib/libctranslate2.so.4.8.1
```

This is installed into:

```text
/usr/local/lib/
```

inside the Docker image.

---

# Automatic Installation

The main AMR Dockerfile automatically installs both artifacts.

The shared library is copied to:

```text
/usr/local/lib/libctranslate2.so.4.8.1
```

and symbolic links are created:

```text
libctranslate2.so.4
libctranslate2.so
```

The system library cache is then refreshed using:

```bash
ldconfig
```

The Python wheel is installed with:

```bash
python3 -m pip install \
    --no-deps \
    --force-reinstall \
    ctranslate2-4.8.1-cp310-cp310-linux_aarch64.whl
```

The Dockerfile removes any previously installed CTranslate2 package before installing this build.

---

# CUDA Verification

The Docker build verifies the installation with:

```bash
python3 -c \
"import ctranslate2; print('CUDA support:', ctranslate2.get_supported_compute_types('cuda'))"
```

The same check can be performed manually inside the container:

```bash
python3 -c \
"import ctranslate2; print(ctranslate2.get_supported_compute_types('cuda'))"
```

A valid response should return supported CUDA compute types rather than a CUDA-unavailable error.

---

# Verify Faster-Whisper

After verifying CTranslate2, test Faster-Whisper:

```bash
python3 - <<'PY'
from faster_whisper import WhisperModel

model = WhisperModel(
    "tiny.en",
    device="cuda",
    compute_type="int8_float16"
)

print("Faster-Whisper CUDA initialization successful")
PY
```

This corresponds to the configuration used by the AMR `stt_node`.

---

# Manual Installation

The Dockerfile is the recommended installation method.

For debugging, the artifacts can also be installed manually inside a compatible Jetson container.

Install the shared library:

```bash
sudo cp lib/libctranslate2.so.4.8.1 /usr/local/lib/
```

Create library links:

```bash
cd /usr/local/lib

sudo ln -sf libctranslate2.so.4.8.1 libctranslate2.so.4
sudo ln -sf libctranslate2.so.4 libctranslate2.so
```

Refresh the dynamic linker cache:

```bash
sudo ldconfig
```

Remove another CTranslate2 Python installation if present:

```bash
python3 -m pip uninstall -y ctranslate2
```

Install the included wheel:

```bash
python3 -m pip install \
    --no-deps \
    --force-reinstall \
    ctranslate2-4.8.1-cp310-cp310-linux_aarch64.whl
```

Verify:

```bash
python3 -c \
"import ctranslate2; print(ctranslate2.__version__); print(ctranslate2.get_supported_compute_types('cuda'))"
```

---

# Why Both Files Are Included

CTranslate2 consists of a native runtime and Python bindings.

For this tested Jetson configuration, the Docker image explicitly installs:

```text
Python wheel
    +
Native CTranslate2 shared library
```

This ensures that the Python package and native GPU runtime used by the project are installed together.

---

# Compatibility

These files are intended for the AMR's tested Jetson environment.

They should not be assumed to work on:

- x86-64 computers
- Windows
- macOS
- incompatible Python versions
- incompatible CUDA environments
- arbitrary JetPack/L4T versions

The wheel name explicitly targets:

```text
cp310-cp310-linux_aarch64
```

so it is intended for **CPython 3.10 on Linux AArch64**.

---

# Upstream Project

CTranslate2 is an open-source inference engine developed by the **OpenNMT** project.

Upstream repository:

```text
https://github.com/OpenNMT/CTranslate2
```

Official installation documentation:

```text
https://opennmt.net/CTranslate2/installation.html
```

CTranslate2 supports optimized Transformer inference and is used by Faster-Whisper for Whisper execution.

---

# Licensing and Attribution

CTranslate2 is distributed by the OpenNMT project under the **MIT License**.

These project-specific binary artifacts should not be interpreted as official OpenNMT release binaries unless they exactly correspond to an official upstream distribution.

When redistributing these files, retain the appropriate CTranslate2 license and attribution.

Upstream license:

```text
https://github.com/OpenNMT/CTranslate2/blob/master/LICENSE
```

For a public repository, it is recommended to keep a copy of the applicable upstream license alongside redistributed third-party binary artifacts.

---

# Role in the AMR

The speech pipeline is:

```text
USB Microphone
      │
      ▼
   STT Node
      │
      ▼
Faster-Whisper
      │
      ▼
CTranslate2
      │
      ▼
Jetson CUDA GPU
      │
      ▼
Recognized Command
      │
      ├──► Decision Node
      │
      └──► LLM / Interaction
```

This allows the AMR to perform local wake-word and command speech recognition while using the Jetson GPU.