FROM python:3.8-slim
WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000
EXPOSE 9000

ENV MODEL_NAME video_resnet_vehicle.VideoResnetVehicle
ENV SERVICE_TYPE MODEL
# ENV FLASK_DEBUG True
ENV TORCH_HOME /app/.torch
# ENV FLASK_SINGLE_THREADED 1
# ENV GUNICORN_WORKERS 0
# ENV GUNICORN_THREADS 0
# ENV GRPC_WORKERS 0
# ENV SELDON_DEBUG 1

RUN chown -R 8888 /app

# CMD exec python cascade_resnet.py
CMD exec seldon-core-microservice $MODEL_NAME --service-type $SERVICE_TYPE