#!/bin/bash

. ~/p3env/bin/activate

set -e
set -o pipefail

if [ -n "$1" ]
then
  if [ -n "$2" ]
  then
    run_on_kitti /home/default/kitti_raw /home/default/kitti_processed /home/default/deeplab_v3_checkpoint/model.ckpt-90000 --day "$1" --start-at "$2"
  else
    run_on_kitti /home/default/kitti_raw /home/default/kitti_processed /home/default/deeplab_v3_checkpoint/model.ckpt-90000 --day "$1"
  fi
else
  run_on_kitti /home/default/kitti_raw /home/default/kitti_processed /home/default/deeplab_v3_checkpoint/model.ckpt-90000
fi

