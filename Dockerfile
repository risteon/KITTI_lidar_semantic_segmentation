FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

# set proxy
ARG DOCKER_PROXY
ENV http_proxy="${DOCKER_PROXY}"
ENV https_proxy="${DOCKER_PROXY}"
ENV HTTP_PROXY="${DOCKER_PROXY}"
ENV HTTPS_PROXY="${DOCKER_PROXY}"
ENV all_proxy="${DOCKER_PROXY}"

# set the device order to match nvidia-smi
ENV CUDA_DEVICE_ORDER="PCI_BUS_ID"

# avoid user interaction when installing tzdata
ENV TZ=Europe/Minsk
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# following https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/dockerfiles/dockerfiles/gpu.Dockerfile
RUN set -ex && \
    # install debian packages
    apt-get update && apt-get install -y --no-install-recommends --allow-downgrades --allow-change-held-packages \
    apt-utils \
    build-essential \
    cuda-command-line-tools-10-0 \
    cuda-cublas-10-0 \
    cuda-cufft-10-0 \
    cuda-curand-10-0 \
    cuda-cusolver-10-0 \
    curl \
    libfreetype6-dev \
    libhdf5-serial-dev \
    libzmq3-dev \
    pkg-config \
    software-properties-common \
    unzip   cuda-cusparse-10-0 \
    software-properties-common \
    ca-certificates \
    libcurl3-gnutls \
    curl \
    wget \
    git \
    cmake \
    make \
    sudo

# install ROS melodic
RUN echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list
COPY ros.key /ros.key
RUN cat /ros.key | apt-key add -

RUN set -ex && \
    apt-get update && apt-get install -y --no-install-recommends \
    ros-melodic-ros-base

# install prerequisities
RUN set -ex && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    libtinyxml-dev \
    liburdfdom-dev \
    liburdfdom-headers-dev \
    liburdfdom-model \
    liburdfdom-model-state \
    liburdfdom-sensor \
    liburdfdom-world \
    libgflags-dev \
    libgflags2.2 \
    libgoogle-glog-dev \
    libamd2 \
    libbtf1 \
    libcamd2 \
    libccolamd2 \
    libcholmod3 \
    libcolamd2 \
    libcxsparse3 \
    libgraphblas1 \
    libklu1 \
    libldl2 \
    libmetis5 \
    librbio2 \
    libspqr2 \
    libsuitesparse-dev \
    libsuitesparseconfig5 \
    libumfpack5 \
    libprotobuf-dev \
    libprotobuf-lite10 \
    libprotobuf10 \
    libprotoc10 \
    protobuf-compiler \
    libprotoc-dev \
    python-alabaster \
    python-babel \
    python-babel-localedata \
    python-certifi \
    python-chardet \
    python-imagesize \
    python-jinja2 \
    python-markupsafe \
    python-pygments \
    python-requests \
    python-sphinx \
    python-typing \
    python-tz \
    python-urllib3 \
    python-wstool \
    python-rosdep \
    sphinx-common \
    sip-dev \
    ninja-build

RUN set -ex && \
    apt-get update && apt-get install -y --no-install-recommends \
    ros-melodic-angles \
    ros-melodic-cv-bridge \
    ros-melodic-eigen-conversions \
    ros-melodic-image-transport \
    ros-melodic-interactive-markers \
    ros-melodic-laser-geometry \
    ros-melodic-map-msgs \
    ros-melodic-media-export \
    ros-melodic-pcl-conversions \
    ros-melodic-python-qt-binding \
    ros-melodic-resource-retriever \
    ros-melodic-tf \
    ros-melodic-tf2 \
    ros-melodic-tf2-eigen \
    ros-melodic-tf2-geometry-msgs \
    ros-melodic-tf2-py \
    ros-melodic-tf2-sensor-msgs \
    ros-melodic-urdf

# install prerequisites for cartographer
RUN set -ex && \
    apt-get install -y --no-install-recommends \
    liblua5.3-dev \
    libcairo2-dev

# install python3 and virtualenv
RUN set -ex && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    virtualenv

# rosdep init
RUN ["/bin/bash", "-c", ". /opt/ros/melodic/setup.bash && rosdep init"]

# enable group sudo without password
RUN set -ex && \
    sed -i -e '/Defaults\s\+env_reset/a Defaults\texempt_group=sudo' /etc/sudoers && \
    sed -i -e 's/%sudo\s*ALL=(ALL:ALL) ALL/%sudo\tALL=(ALL) NOPASSWD:ALL/g' /etc/sudoers

# switch to non-root installation. Use group with specific ID to manage permissions on host filesystem.
RUN useradd -ms /bin/bash default
RUN addgroup --gid 85200 wgroup
RUN adduser default sudo
RUN adduser default wgroup
USER default
WORKDIR /home/default

# build google cartographer
COPY --chown=default catkin_ws catkin_ws
COPY --chown=default kitti2bag kitti2bag
COPY --chown=default read_transform read_transform
# remove cartographer rviz (not necessary)
RUN \
    rm -rf ~/catkin_ws/src/cartographer_ros/cartographer_rviz

WORKDIR /home/default
# create virtualenv with python2 for cartographer
RUN virtualenv p2env
RUN \
    set -ex; \
    . p2env/bin/activate && \
    pip install --no-cache-dir catkin_pkg pyyaml empy opencv-python pykitti rospkg pycrypto gnupg && \
    pip install --no-cache-dir ~/kitti2bag && \
    pip install --no-cache-dir ~/read_transform

RUN /bin/bash -c \
    ". p2env/bin/activate && . /opt/ros/melodic/setup.bash && \
    cd ~/catkin_ws/src && catkin_init_workspace && \
    cd ~/catkin_ws && src/cartographer/scripts/install_proto3.sh"

RUN /bin/bash -c \
    ". p2env/bin/activate && . /opt/ros/melodic/setup.bash && \
    cd ~/catkin_ws && \
    catkin_make_isolated --install --use-ninja && \
    rm -rf ~/catkin_ws/build_isolated"

# create virtual env for python3 with tensorflow
RUN python3 -m venv p3env
RUN \
    set -ex; \
    . p3env/bin/activate && \
    pip install --no-cache-dir wheel && \
    pip install --no-cache-dir tensorflow-gpu==1.13.1

# copy tensorflow models
COPY --chown=default models models
# install models and delete unused folders that are large in size
RUN \
    set -ex; \
    . p3env/bin/activate && \
    pip install --no-cache-dir -e ~/models/research && \
    pip install --no-cache-dir -e ~/models/research/slim && \
    rm -rf ~/models/research/gan ~/models/research/syntaxnet ~/models/research/object_detection

# copy and install python code for semantics generation and deeplab checkpoint
COPY --chown=default deeplab_v3_checkpoint deeplab_v3_checkpoint
COPY --chown=default kitti_lidar_semantics kitti_lidar_semantics
RUN \
    set -ex; \
    . p3env/bin/activate && \
    pip install --no-cache-dir -e ~/kitti_lidar_semantics

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV CARTOGRAPHER_CATKIN_WS="/home/default/catkin_ws"
ENV CARTOGRAPHER_ENV="/home/default/p2env/"

# copy script folder
COPY --chown=default script script
ENTRYPOINT ["/bin/bash", "/home/default/script/process_kitti.sh"]
CMD []

