#!/bin/bash

echo "============================================="
echo "🤖 Starting AMR Robot Autonomous Stack 🤖"
echo "============================================="

WEB_PID=""

cleanup() {
    echo ""
    echo "🛑 Shutting down host Web Server..."

    if [[ -n "$WEB_PID" ]] && kill -0 "$WEB_PID" 2>/dev/null; then
        kill "$WEB_PID" 2>/dev/null
        wait "$WEB_PID" 2>/dev/null
    fi

    # Restore X access permission added earlier
    xhost -local:docker >/dev/null 2>&1

    # Keep the persistent container running
    # docker stop amr_base_container

    echo "✅ Cleanup complete."
}

trap cleanup INT TERM EXIT

# 1. Enable GUI forwarding permissions for Docker
xhost +local:docker

# 2. Start the local Web UI server
echo "🌐 Launching local webpage on port 8000..."

cd "$HOME/projects/my_robot/robot_ui" || {
    echo "❌ Web UI folder not found!"
    exit 1
}

python3 -m http.server 8000 >/dev/null 2>&1 &
WEB_PID=$!

cd "$HOME" || exit 1

# 3. Start the persistent Docker container
echo "🐳 Waking up container: amr_base_container..."

if ! docker start amr_base_container >/dev/null; then
    echo "❌ Failed to start Docker container!"
    exit 1
fi

# 4. Launch the ROS 2 stack inside the container
echo "🚀 Launching all autonomous nodes..."

docker exec -it amr_base_container /bin/bash -c '
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    export TTS_FFMPEG_THREADS=1

    source /opt/ros/humble/setup.bash
    source /workspace/install/setup.bash

    echo "Thread limits:"
    echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
    echo "  OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS"
    echo "  MKL_NUM_THREADS=$MKL_NUM_THREADS"
    echo "  NUMEXPR_NUM_THREADS=$NUMEXPR_NUM_THREADS"
    echo "  TTS_FFMPEG_THREADS=$TTS_FFMPEG_THREADS"

    exec ros2 launch robot_bringup my_robot.launch.py
'
