#!/bin/bash
echo "============================================="
echo "🤖 Starting AMR Robot Autonomous Stack 🤖"
echo "============================================="
# 1. Enable GUI forward permissions for Docker
xhost +local:docker
# 2. Navigate to your Web UI folder and spin up the Python server in the background
echo "🌐 Launching local webpage on port 8000..."
cd ~/projects/my_robot/robot_ui || { echo "❌ Web UI folder not found!"; exit 1; }
python3 -m http.server 8000 > /dev/null 2>&1 &
WEB_PID=$!
cd ~/   # or wherever your workspace root is, for later commands
# 3. Start your existing, persistent Docker container
echo "🐳 Waking up container: my_robotics_env..."
docker start my_robotics_env
# 4. Set up cleanup BEFORE the blocking foreground command
function ctrl_c() {
    echo ""
    echo "🛑 Shutting down host Web Server..."
    kill $WEB_PID 2>/dev/null
    # echo "🐳 Stopping Docker container..."
    # docker stop my_robotics_env
    exit 0
}
trap ctrl_c INT
# 5. Execute your main ROS 2 launch command inside the running container
echo "🚀 Launching all autonomous nodes..."
docker exec -it my_robotics_env /bin/bash -c "
    source /opt/ros/humble/setup.bash;
    source install/setup.bash;
    ros2 launch robot_bringup my_robot.launch.py
"
# In case docker exec exits on its own without Ctrl+C, still clean up
ctrl_c