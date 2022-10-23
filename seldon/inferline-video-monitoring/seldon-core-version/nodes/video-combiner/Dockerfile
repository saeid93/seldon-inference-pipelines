FROM python:3.7-slim
WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000
EXPOSE 9000

ENV MODEL_NAME video_combiner.VideoCombiner
ENV SERVICE_TYPE COMBINER
# ENV FLASK_DEBUG True
# ENV SELDON_LOG_LEVEL DEBUG

RUN chown -R 8888 /app

CMD exec seldon-core-microservice $MODEL_NAME --service-type $SERVICE_TYPE