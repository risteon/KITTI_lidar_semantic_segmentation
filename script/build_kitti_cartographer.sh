#!/usr/bin/env bash
set -e
set -o pipefail

virtualenv p2env
. p2env/bin/activate
. /opt/ros/melodic/setup.bash

set -u

cd ~/catkin_ws/src
catkin_init_workspace
cd ~/catkin_ws
src/cartographer/scripts/install_proto3.sh
#rosdep update
#rosdep install --from-paths src --ignore-src --rosdistro=${ROS_DISTRO} -y
catkin_make_isolated --install --use-ninja

