import os
import time
import json
import psutil
from mlserver import MLModel
import numpy as np
from mlserver.codecs import NumpyCodec
from mlserver.logging import logger
from mlserver.utils import get_model_uri
from mlserver.types import (
    InferenceRequest,
    InferenceResponse,
    ResponseOutput,
    Parameters)
from mlserver import MLModel
from mlserver.codecs import DecodedParameterName
from mlserver.cli.serve import load_settings
from copy import deepcopy
from transformers import pipeline
from mlserver.codecs import StringCodec
from mlserver_huggingface.common import NumpyEncoder
from typing import List, Dict


try:
    PREDICTIVE_UNIT_ID = os.environ['PREDICTIVE_UNIT_ID']
    logger.error(f'PREDICTIVE_UNIT_ID set to: {PREDICTIVE_UNIT_ID}')
except KeyError as e:
    PREDICTIVE_UNIT_ID = 'predictive_unit'
    logger.error(
        f"PREDICTIVE_UNIT_ID env variable not set, using default value: {PREDICTIVE_UNIT_ID}")

class GeneralAudio(MLModel):
    async def load(self):
        self.loaded = False
        self.request_counter = 0
        self.batch_counter = 0
        try:
            self.MODEL_VARIANT = os.environ['MODEL_VARIANT']
            logger.error(f'MODEL_VARIANT set to: {self.MODEL_VARIANT}')
        except KeyError as e:
            self.MODEL_VARIANT = 'facebook/s2t-small-librispeech-asr'
            logger.error(
                f"MODEL_VARIANT env variable not set, using default value: {self.MODEL_VARIANT}")
        try:
            self.TASK = os.environ['TASK']
            logger.error(f'TASK set to: {self.TASK}')
        except KeyError as e:
            self.TASK = 'automatic-speech-recognition' 
            logger.error(
                f"TASK env variable not set, using default value: {self.TASK}")
        logger.info('Loading the ML models')
        # TODO add batching like the runtime
        logger.error(f'max_batch_size: {self._settings.max_batch_size}')
        logger.error(f'max_batch_time: {self._settings.max_batch_time}')
        # self.model  = lambda l: l
        self.model = pipeline(
            task=self.TASK,
            model=self.MODEL_VARIANT,
            batch_size=self._settings.max_batch_size)
        self.loaded = True
        # logger.info('model loading complete!')
        # container_total_memory = psutil.virtual_memory()[1]/ (10**6)
        # logger.error(f"container_total_memory: {container_total_memory}")
        # container_total_cpu = psutil.cpu_count()
        # logger.error(f"container_total_cpu: {container_total_cpu}")
        # cpu_usage_percentage = psutil.cpu_percent(1)
        # logger.error(f'CPU % used: {cpu_usage_percentage}')
        # total_memory, used_memory, _ = map(
        #     int, os.popen('free -t -m').readlines()[-1].split()[1:])
        # memory_usage_percentage = round((used_memory/total_memory) * 100, 2)
        # logger.error(f"RAM % used: {memory_usage_percentage}")
        return self.loaded

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        if self.loaded == False:
            self.load()
        # logger.error(f"payload:\n{payload}")
        arrival_time = time.time()
        for request_input in payload.inputs:
            logger.error('request input shape:\n')
            logger.error(f"{request_input.shape}\n")
            decoded_input = self.decode(request_input)
            logger.error(decoded_input)
            X = decoded_input
        X = list(map(lambda l: np.array(l), X))
        received_batch_len = len(X)
        logger.error(f"recieved batch len:\n{received_batch_len}")
        self.request_counter += received_batch_len
        self.batch_counter += 1
        logger.error(f"to the model:\n{type(X)}")
        logger.error(f"type of the to the model:\n{type(X)}")
        logger.error(f"len of the to the model:\n{len(X)}")
        output: List[Dict] = self.model(X)
        logger.error(f"model output:\n{output}")
        serving_time = time.time()
        timing = {
            f"arrival_{PREDICTIVE_UNIT_ID}".replace("-","_"): arrival_time,
            f"serving_{PREDICTIVE_UNIT_ID}".replace("-", "_"): serving_time
        }

        # cpu_usage_percentage = psutil.cpu_percent(1)
        # logger.error(f'CPU usage is: {cpu_usage_percentage}')
        # total_memory, used_memory, _ = map(
        #     int, os.popen('free -t -m').readlines()[-1].split()[1:])
        # memory_usage_percentage = round((used_memory/total_memory) * 100, 2)
        # logger.error(f"RAM memory % used: {memory_usage_percentage}")


        output_with_time = list()
        for pred in output:
            output_with_time.append(
                {
                    'time': timing,
                    'output': pred,                
                }
            )
        str_out = [json.dumps(
            pred, cls=NumpyEncoder) for pred in output_with_time]
        prediction_encoded = StringCodec.encode_output(
            payload=str_out, name="output")
        logger.error(f"Output:\n{prediction_encoded}\nwas sent!")
        logger.error(f"request counter:\n{self.request_counter}\n")
        logger.error(f"batch counter:\n{self.batch_counter}\n")
        return InferenceResponse(
            id=payload.id,
            model_name=self.name,
            model_version=self.version,
            outputs = [prediction_encoded]
        )
