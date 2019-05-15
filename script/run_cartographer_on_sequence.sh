#!/usr/bin/env bash
kitti_root="$1"
tmp_dir="$2"
kitti_day="$3"
kitti_s="$4"
kitti_output="$5"
echo "KITTI ROOT: ${kitti_root}"
echo "KITTI DAY: ${kitti_day}"
echo "KITTI SEQ: ${kitti_s}"
echo "TMP DIR: ${tmp_dir}"
echo "KITTI OUTPUT: ${kitti_output}"

. "~/p2env/bin/activate"
source /opt/ros/melodic/setup.bash

# after conda source because activate fails without $PS1 in bash strict mode
set -eu
set -o pipefail

cd ~/catkin_ws
source install_isolated/setup.bash
cd ${tmp_dir}
kitti2bag -t ${kitti_day} -r ${kitti_s} --dir ${tmp_dir} raw_synced ${kitti_root} 

cd /lhome/chrrist/workspace/cartographer/catkin_ws/install_isolated/bin
# run cartographer in offline mode
roslaunch cartographer_kitti offline_kitti_no_rviz.launch bag_filenames:=${tmp_dir}/kitti_${kitti_day}_drive_${kitti_s}_synced.bag

# convert pbstream to bagfile
cd /lhome/chrrist/workspace/cartographer/catkin_ws
cartographer_dev_pbstream_trajectories_to_rosbag -input ${tmp_dir}/kitti_${kitti_day}_drive_${kitti_s}_synced.bag.pbstream -output ${tmp_dir}/kitti_poses.bag
# 
mkdir -p ${kitti_output}/poses_cartographer
read_transform_from_rosbag ${tmp_dir}/kitti_poses.bag ${kitti_output}/poses_cartographer/timestamps.txt ${kitti_output}/poses_cartographer/poses.txt

