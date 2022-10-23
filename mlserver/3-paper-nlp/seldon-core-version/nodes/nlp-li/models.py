import os
import time
import json
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

class GeneralNLP(MLModel):
    async def load(self):
        self.loaded = False
        self.request_counter = 0
        self.batch_counter = 0
        try:
            self.MODEL_VARIANT = os.environ['MODEL_VARIANT']
            logger.error(f'MODEL_VARIANT set to: {self.MODEL_VARIANT}')
        except KeyError as e:
            self.MODEL_VARIANT = 'dinalzein/xlm-roberta-base-finetuned-language-identification'
            logger.error(
                f"MODEL_VARIANT env variable not set, using default value: {self.MODEL_VARIANT}")
        try:
            self.TASK = os.environ['TASK']
            logger.error(f'TASK set to: {self.TASK}')
        except KeyError as e:
            self.TASK = 'text-classification' 
            logger.error(
                f"MODEL_VARIANT env variable not set, using default value: {self.TASK}")
        logger.error('Loading the ML models')
        # TODO add batching like the runtime
        logger.error(f'max_batch_size: {self._settings.max_batch_size}')
        logger.error(f'max_batch_time: {self._settings.max_batch_time}')
        self.model  = pipeline(
            task=self.TASK,
            model=self.MODEL_VARIANT,
            batch_size=self._settings.max_batch_size)
        self.loaded = True
        logger.error('model loading complete!')
        return self.loaded

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        if self.loaded == False:
            self.load()
        logger.error(f"payload:\n{payload}")
        arrival_time = time.time()
        for request_input in payload.inputs:
            logger.error('request input:\n')
            logger.error(f"{request_input}\n")
            decoded_input = self.decode(request_input)
            logger.error(decoded_input)
            X = decoded_input
        X = list(X)
        logger.error(f"to the model:\n{X}")
        logger.error(f"type of the to the model:\n{type(X)}")
        logger.error(f"len of the to the model:\n{len(X)}")
        received_batch_len = len(X)
        logger.error(f"recieved batch len:\n{received_batch_len}")
        self.request_counter += received_batch_len
        self.batch_counter += 1
        output: List[Dict] = self.model(X)
        logger.error(f"model output:\n{output}")
        serving_time = time.time()
        timing = {
            f"arrival_{PREDICTIVE_UNIT_ID}".replace("-","_"): arrival_time,
            f"serving_{PREDICTIVE_UNIT_ID}".replace("-", "_"): serving_time
        }
        output_with_time = list()
        for input, pred in zip(X, output):
            output = {
                'label': pred['label'],
                'input': input
            }
            output_with_time.append(
                {
                    'time': timing,
                    'output': output,
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
