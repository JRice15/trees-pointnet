#!/bin/bash

# Runs arbitrary command in the docker container
# Sample usage:
#   ./docker_run.sh <GPU> python train.py
# where <GPU> is an integer number specifying the GPU to make visible in the container

[ "$1" -ge 0 ] || { echo "supply a non-negative gpu number" && exit 1; }

docker run -u $(id -u):$(id -g) --rm -it \
    -v /data:/data \
    --runtime=nvidia \
    -e CUDA_VISIBLE_DEVICES="$1" \
    -v /home/$USER:/home/$USER \
    -e USER=$USER \
    -w $PWD \
    trees2.2.2 "${@:2}"


