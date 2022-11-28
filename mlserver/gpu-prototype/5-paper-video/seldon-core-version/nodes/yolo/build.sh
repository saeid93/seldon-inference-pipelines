rm -r __pycache__
rm *.pt
REPOS=(
    sdghafouri)
IMAGE_NAME=video-pipelines-mlserver-gpu:yolo
mlserver build --tag=$IMAGE_NAME .
for REPO in ${REPOS[@]}
do
    docker tag $IMAGE_NAME $REPO/$IMAGE_NAME
    docker push $REPO/$IMAGE_NAME
done