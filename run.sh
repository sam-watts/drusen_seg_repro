docker build -t drusen-seg-inference . && \
DOCKER_BUILDKIT=1 docker run \
    --gpus 0 \
    -v ${PWD}/../outputs:/opt/outputs \
    -v ${PWD}/../data:/opt/data \
    drusen-seg-inference \
