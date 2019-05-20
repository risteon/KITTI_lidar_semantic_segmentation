#!/bin/bash
mkdir "$1" && chown :85200 "$1" && chmod 775 "$1" && chmod g+s "$1"
docker run --runtime=nvidia --mount type=bind,source=/mapr/738.mbc.de/data/training/datasets/rd/athena/atplid/kitti_raw,target=/home/default/kitti_raw,readonly --mount type=bind,source="$1",target=/home/default/kitti_processed kitti_lidar_semantics:base 2011_09_29

