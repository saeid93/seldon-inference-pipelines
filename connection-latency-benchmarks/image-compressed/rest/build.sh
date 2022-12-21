rm -r __pycache__
rm *.pt
REPOS=(
    sdghafouri
    gcr.io/hale-ivy-335012)
IMAGE_NAME=latency-connection:rest-image-base64
mlserver build --tag=$IMAGE_NAME .
for REPO in ${REPOS[@]}
do
    docker tag $IMAGE_NAME $REPO/$IMAGE_NAME
    docker push $REPO/$IMAGE_NAME
done