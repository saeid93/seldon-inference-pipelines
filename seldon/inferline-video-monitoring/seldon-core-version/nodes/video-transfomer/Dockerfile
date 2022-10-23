FROM python:3.7-slim
WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
# RUN apt-get update
# RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY . .

EXPOSE 5000
EXPOSE 9000

ENV MODEL_NAME video_transformer.VideoTransformer
ENV SERVICE_TYPE TRANSFORMER
# ENV FLASK_DEBUG True
# ENV SELDON_LOG_LEVEL DEBUG

RUN chown -R 8888 /app

CMD exec seldon-core-microservice $MODEL_NAME --service-type $SERVICE_TYPE