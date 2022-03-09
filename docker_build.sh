#!/bin/bash

echo
echo "Building docker image"
docker build envs/ -t trees2.2.2

echo 
echo "Compiling TensorFlow ops"
./docker_run.sh 3 ./compile_ops.sh
