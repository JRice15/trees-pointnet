#!/bin/bash

[ "$1" -ge 0 ] || { echo "supply a non-negative gpu number" && exit 1; }

docker run -u $(id -u):$(id -g) --rm -it \
    -v /data:/data \
    --runtime=nvidia \
    -e CUDA_VISIBLE_DEVICES="$1" \
    -v /home/$USER:/home/$USER \
    -e USER=$USER \
    -w $PWD \
    trees2.2.2 "${@:2}"


