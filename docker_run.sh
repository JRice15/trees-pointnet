
if [ "$1" = "" ]; then
    echo "supply a gpu number"
    exit 1
fi

docker run -u $(id -u):$(id -g) --rm -it \
    -v /data:/data \
    --runtime=nvidia \
    -e CUDA_VISIBLE_DEVICES="$1" \
    -v /home/$USER:/home/$USER \
    -e USER=$USER \
    -w $PWD \
    trees2.2.2 bash


