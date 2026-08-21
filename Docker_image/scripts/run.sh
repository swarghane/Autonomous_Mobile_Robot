#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

IMAGE_NAME="amr_base_image:latest"
CONTAINER_NAME="amr_base_container"
HOST_UID="$(id -u)"
PULSE_DIR="/run/user/${HOST_UID}/pulse"

if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo "❌ Docker image '$IMAGE_NAME' not found."
    echo "Build it first with:"
    echo "  ./Docker_image/scripts/build.sh"
    exit 1
fi

if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    echo "🐳 Container '$CONTAINER_NAME' already exists."

    if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
        echo "▶️ Starting existing container..."
        docker start "$CONTAINER_NAME" >/dev/null
    fi

    exec docker exec -it "$CONTAINER_NAME" bash
fi

echo "🐳 Creating container: $CONTAINER_NAME"
echo "📁 Workspace: $REPO_ROOT → /workspace"

docker run -dit \
    --name "$CONTAINER_NAME" \
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
    -v "$PULSE_DIR:$PULSE_DIR" \
    -v "$HOME/.config/pulse/cookie:/root/.config/pulse/cookie" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /tmp/argus_socket:/tmp/argus_socket \
    -v "$REPO_ROOT:/workspace" \
    -w /workspace \
    -e DISPLAY="$DISPLAY" \
    -e QT_X11_NO_MITSHM=1 \
    -e PULSE_SERVER="unix:$PULSE_DIR/native" \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    "$IMAGE_NAME"

echo "✅ Container created."

exec docker exec -it "$CONTAINER_NAME" bash