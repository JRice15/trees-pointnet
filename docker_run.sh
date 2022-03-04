
if [ "$1" = "" ]; then
    echo "supply a gpu number"
    exit 1
fi

#docker pull tensorflow/tensorflow:2.2.2-gpu-py3
#docker run --rm -it --gpus $1 -w $PWD -v /data:/data -e USER=$USER -e HOME=/data/$USER tensorflow/tensorflow:2.2.2-gpu-py3 bash

docker run -u $(id -u):$(id -g) --rm -it -v /data:/data -v /home/$USER:/home/$USER \
    --runtime=nvidia -e CUDA_VISIBLE_DEVICES="$1" -e USER=$USER -w $PWD \
    trees2.2.2 bash

