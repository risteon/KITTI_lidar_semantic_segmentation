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

# following https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/dockerfiles/dockerfiles/gpu.Dockerfile
RUN set -ex && \
    # install debian packages
    apt-get update && apt-get install -y --no-install-recommends --allow-downgrades --allow-change-held-packages \
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
    make

# install ROS melodic
RUN echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list
COPY ros.key /ros.key
RUN cat /ros.key | apt-key add -
# avoid user interaction when installing tzdata
ENV TZ=Europe/Minsk
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update
RUN set -ex && \
    apt-get install -y --no-install-recommends \
    ros-melodic-ros-base

# install python3 and virtualenv
RUN set -ex && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    virtualenv

# switch to non-root installation
RUN useradd -ms /bin/bash default
USER default
WORKDIR /home/default
# create virtual env for python3 with tensorflow
RUN python3 -m venv p3env
RUN \
    set -ex; \
    . p3env/bin/activate && \
    pip install tensorflow-gpu==1.13.1

# copy data and repositories
COPY script script
COPY deeplab_v3_checkpoint deeplab_v3_checkpoint
COPY kitti_lidar_semantics kitti_lidar_semantics
COPY catkin_ws catkin_ws

# build google cartographer
WORKDIR /home/default/catkin_ws/src
RUN catkin_init_workspace

