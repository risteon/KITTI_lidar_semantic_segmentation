```
$ sudo docker build --build-arg DOCKER_PROXY=http://172.17.0.1:33128 -t kitti_lidar_semantics:base .
$ sudo docker run -it --runtime=nvidia kitti_lidar_semantics:base
```
