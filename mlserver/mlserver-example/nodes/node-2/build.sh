REPOS=(
    sdghafouri
    gcr.io/hale-ivy-335012)
IMAGE_NAME=custom-mlserver:node-2
mlserver build . -t $IMAGE_NAME
for REPO in ${REPOS[@]}
do
    docker tag $IMAGE_NAME $REPO/$IMAGE_NAME
    docker push $REPO/$IMAGE_NAME
done