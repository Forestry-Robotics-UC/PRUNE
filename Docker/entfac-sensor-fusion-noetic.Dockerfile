FROM ros:noetic-ros-core

ARG USERNAME=ros
ARG USER_UID=1000
ARG USER_GID=1000
ARG UBUNTU_APT_MIRROR=http://de.archive.ubuntu.com/ubuntu

# Install ROS dependencies and basic tooling
RUN sed -i "s|http://archive.ubuntu.com/ubuntu|${UBUNTU_APT_MIRROR}|g" /etc/apt/sources.list && \
    apt-get -o Acquire::Retries=10 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 update && \
    DEBIAN_FRONTEND=noninteractive apt-get -o Acquire::Retries=10 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 install -y --fix-missing --no-install-recommends \
        sudo \
        build-essential \
        cmake \
        python3-pip \
        python3-numpy \
        python3-rospkg \
        python3-catkin-tools \
        python3-rosdep \
        python3-scipy \
        ros-noetic-catkin \
        ros-noetic-tf2-ros \
        ros-noetic-message-filters \
        ros-noetic-sensor-msgs \
        ros-noetic-std-msgs \
        ros-noetic-image-transport \
        ros-noetic-compressed-image-transport \
        ros-noetic-compressed-depth-image-transport \
        ros-noetic-robot-state-publisher \
        ros-noetic-xacro \
        ros-noetic-rosbag \
        ros-noetic-roslaunch \
        ros-noetic-rviz \
    && rm -rf /var/lib/apt/lists/*

# Initialize rosdep (update may require network; run it at runtime if needed)
RUN rosdep init || true

# Optional: pytest for quick checks
RUN pip3 install --no-cache-dir pytest

# Create non-root user
RUN groupadd --gid ${USER_GID} ${USERNAME} && \
    useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME}

# Initialize workspace
USER ${USERNAME}
WORKDIR /home/${USERNAME}/ws
RUN mkdir -p /home/${USERNAME}/ws/src

# Shell conveniences: always source ROS, and source the workspace if built.
RUN echo "source /opt/ros/noetic/setup.bash" >> /home/${USERNAME}/.bashrc && \
    echo "if [ -f /home/${USERNAME}/ws/devel/setup.bash ]; then source /home/${USERNAME}/ws/devel/setup.bash; fi" >> /home/${USERNAME}/.bashrc

# Best-effort rosdep update (may fail in offline environments)
RUN rosdep update || true

# Default to interactive shell with ROS env sourced
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
