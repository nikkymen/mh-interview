NAME=openface
REGISTRY=nikkymen

docker rm -f ${NAME}
docker build -t ${REGISTRY}/${NAME} -f docker/Dockerfile .
