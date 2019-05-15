#!/usr/bin/env bash
set -eu
set -o pipefail

virtualenv p2env
. p2env/bin/activate
. /opt/ros/melodic/setup.bash

