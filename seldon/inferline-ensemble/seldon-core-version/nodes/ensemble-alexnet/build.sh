IMAGE_NAME=inferline:ensemble-alexnet && \
docker build --tag=$IMAGE_NAME . && \
docker tag $IMAGE_NAME sdghafouri/$IMAGE_NAME && \
docker push sdghafouri/$IMAGE_NAME
REPOS=(
    sdghafouri
    gcr.io/hale-ivy-335012)
IMAGE_NAME=inferline:ensemble-alexnet
docker build --tag=$IMAGE_NAME .
for REPO in ${REPOS[@]}
do
    docker tag $IMAGE_NAME $REPO/$IMAGE_NAME
    docker push $REPO/$IMAGE_NAME
done