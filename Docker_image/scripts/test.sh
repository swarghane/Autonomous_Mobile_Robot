#!/bin/bash

python3 -c "import torch; print(torch.cuda.is_available())"

python3 -c "import ctranslate2; print(ctranslate2.get_supported_compute_types('cuda'))"

python3 -c "from faster_whisper import WhisperModel; print('Whisper OK')"

source /opt/ros/humble/setup.bash

ros2 doctor