rm -r __pycache__
REPOS=(
sdghafouri)
    # gcr.io/hale-ivy-335012)
IMAGE_NAME=sum-qa-pipelines-mlserver:nlpqa
mlserver build . -t $IMAGE_NAME
for REPO in ${REPOS[@]}
do
    docker tag $IMAGE_NAME $REPO/$IMAGE_NAME
    docker push $REPO/$IMAGE_NAME
done