docker build -t drusen-seg-inference . && \
DOCKER_BUILDKIT=1 docker run drusen-seg-inference \
    baseline \
    -v $PWD/outputs:/outputs \
    -v $PWD/data:/data \
