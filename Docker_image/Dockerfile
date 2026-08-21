FROM dustynv/l4t-pytorch:r36.4.0

ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]

# -----------------------------------------------------------------------------
# Remove inherited Jetson pip configuration
# -----------------------------------------------------------------------------
RUN sed -i '/^PIP_/d' /etc/environment || true

ENV PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_EXTRA_INDEX_URL=
ENV PIP_TRUSTED_HOST="pypi.org files.pythonhosted.org"
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_ROOT_USER_ACTION=ignore
ENV PYTHONUNBUFFERED=1

RUN mkdir -p /root/.config/pip && \
    printf '%s\n' \
    '[global]' \
    'index-url = https://pypi.org/simple' \
    'no-cache-dir = true' \
    'disable-pip-version-check = true' \
    '' \
    'trusted-host =' \
    '    pypi.org' \
    '    files.pythonhosted.org' \
    > /root/.config/pip/pip.conf

# -----------------------------------------------------------------------------
# Locale
# -----------------------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        gnupg2 \
        lsb-release \
        locales && \
    locale-gen en_US.UTF-8 && \
    update-locale LANG=en_US.UTF-8 && \
    rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# -----------------------------------------------------------------------------
# ROS 2 Repository
# -----------------------------------------------------------------------------
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    | gpg --dearmor \
    -o /usr/share/keyrings/ros-archive-keyring.gpg

RUN echo "deb [arch=arm64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu jammy main" \
    > /etc/apt/sources.list.d/ros2.list

# -----------------------------------------------------------------------------
# Install Ubuntu + ROS packages
# -----------------------------------------------------------------------------
COPY apt_packages.txt /tmp/apt_packages.txt

RUN apt-get update && \
    grep -vE '^\s*#|^\s*$' /tmp/apt_packages.txt | \
    xargs apt-get install -y --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Initialize rosdep
# -----------------------------------------------------------------------------
RUN rosdep init || true

# -----------------------------------------------------------------------------
# Workspace
# -----------------------------------------------------------------------------
WORKDIR /workspace

# -----------------------------------------------------------------------------
# Python packages
# -----------------------------------------------------------------------------
COPY requirements.txt .

RUN python3 -m pip config debug && \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install -r requirements.txt

# -----------------------------------------------------------------------------
# Install GPU-enabled CTranslate2 shared library
# -----------------------------------------------------------------------------
COPY wheels/ctranslate2_gpu/lib/libctranslate2.so.4.8.1 /usr/local/lib/

RUN cd /usr/local/lib && \
    ln -sf libctranslate2.so.4.8.1 libctranslate2.so.4 && \
    ln -sf libctranslate2.so.4 libctranslate2.so && \
    ldconfig

# -----------------------------------------------------------------------------
# Install GPU-enabled CTranslate2 wheel
# -----------------------------------------------------------------------------
COPY wheels/ctranslate2_gpu/ctranslate2-4.8.1-cp310-cp310-linux_aarch64.whl /tmp/

RUN python3 -m pip uninstall -y ctranslate2 || true && \
    python3 -m pip install \
        --no-deps \
        --force-reinstall \
        /tmp/ctranslate2-4.8.1-cp310-cp310-linux_aarch64.whl && \
    python3 -c "import ctranslate2; print('CUDA support:', ctranslate2.get_supported_compute_types('cuda'))"

# -----------------------------------------------------------------------------
# Clean temporary files
# -----------------------------------------------------------------------------
RUN rm -f /tmp/apt_packages.txt \
          /tmp/requirements.txt \
          /tmp/ctranslate2-4.8.1-cp310-cp310-linux_aarch64.whl
RUN rm -rf /root/.cache/pip

# -----------------------------------------------------------------------------
# ROS Environment
# -----------------------------------------------------------------------------
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

CMD ["bash"]