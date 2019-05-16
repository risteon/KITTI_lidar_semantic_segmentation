#!/bin/bash

. ~/p3env/bin/activate

set -eu
set -o pipefail

run_on_kitti /home/default/kitti_raw /home/default/kitti_processed /home/default/deeplab_v3_checkpoint/model.ckpt-90000

