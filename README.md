# Make sure submodules are checked out
```
git submodule init && git submodule update
```

# Use the provided Dockerfile
```
$ sudo docker build --build-arg DOCKER_PROXY=http://<DOCKER_IP>:<PORT> -t kitti_lidar_semantics:base .
$ mkdir kitti_output && sudo chown :85200 kitti_output && sudo chmod 775 kitti_output && sudo chmod g+s kitti_output
$ sudo docker run -it \
    --runtime=nvidia \
    --mount type=bind,source=<PATH_TO_KITTI_RAW>,target=/home/default/kitti_raw,readonly \
    --mount type=bind,source=<PATH_TO_OUTPUT_FOLDER>,target=/home/default/kitti_processed \
    kitti_lidar_semantics:base
```

# Optional: specify day to process and sequence number to start from
```
$ ... 2011_09_29 0011
```
