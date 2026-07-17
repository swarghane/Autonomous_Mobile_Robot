FROM dustynv/l4t-pytorch:r36.4.0

ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]

# Install ROS2 repository tools
RUN apt-get update && apt-get install -y \
    curl \
    gnupg2 \
    lsb-release \
    software-properties-common \
    locales

RUN locale-gen en_US en_US.UTF-8

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# Add ROS2 repository
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | apt-key add -

RUN echo "deb http://packages.ros.org/ros2/ubuntu jammy main" \
    > /etc/apt/sources.list.d/ros2.list

# Install ROS2 + important robotics packages
RUN apt-get update && apt-get install -y \
    ros-humble-ros-base \
    python3-colcon-common-extensions \
    python3-rosdep \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    ros-humble-vision-msgs \
    ros-humble-tf2-ros \
    ros-humble-geometry-msgs \
    ros-humble-sensor-msgs \
    ros-humble-web-video-server \
    portaudio19-dev \
    ffmpeg \
    espeak-ng \
    git \
    nano \
    wget \
    iputils-ping

WORKDIR /workspace

# Install Python libraries
COPY requirements.txt .

ENV PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_TRUSTED_HOST="pypi.org files.pythonhosted.org"

RUN pip3 install --no-cache-dir -r requirements.txt

CMD ["bash"]