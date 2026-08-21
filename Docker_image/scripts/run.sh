#!/bin/bash
    
docker run -it \
    --name amr_base_container \
    --runtime nvidia \
    --network host \
    --ipc host \
    --privileged \
    --pid host \
    --restart no \
    --device /dev/video0 \
    --device /dev/video1 \
    --device /dev/snd \
    --device /dev/bus/usb \
    --device /dev/i2c-0 \
    --device /dev/i2c-1 \
    --device /dev/ttyUSB0 \
    --device /dev/ttyACM0 \
    -v /dev:/dev \
    -v /run/udev:/run/udev:ro \
    -v /run/user/1000/pulse:/run/user/1000/pulse \
    -v ~/.config/pulse/cookie:/root/.config/pulse/cookie \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /tmp/argus_socket:/tmp/argus_socket \
    -v $(pwd):/workspace \
    -w /workspace \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -e PULSE_SERVER=unix:/run/user/1000/pulse/native \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    amr_base_image:latest
