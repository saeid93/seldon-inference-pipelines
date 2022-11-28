#!/bin/bash

REPOS=(
    sdghafouri)
IMAGE_NAME=audio-sent-pipelines-mlserver-gpu:nlpsent
PYTHON_ENV=central
mlserver build . -t $IMAGE_NAME
for REPO in ${REPOS[@]}
do
    docker tag $IMAGE_NAME $REPO/$IMAGE_NAME
    docker push $REPO/$IMAGE_NAME
done