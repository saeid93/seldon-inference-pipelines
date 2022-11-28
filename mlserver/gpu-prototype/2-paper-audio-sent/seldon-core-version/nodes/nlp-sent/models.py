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

def to_bool(x):
    return x in ("True", "true", True)

try:
    GPU = to_bool(os.environ['GPU'])
    logger.error(f'PREDICTIVE_UNIT_ID set to: {GPU}')
except KeyError as e:
    GPU = False
    logger.error(
        f"GPU env variable not set, using default value: {GPU}")

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
            self.TASK = 'sentiment-analysis' 
            logger.error(
                f"MODEL_VARIANT env variable not set, using default value: {self.TASK}")
        logger.error('Loading the ML models')
        # TODO add batching like the runtime
        logger.error(f'max_batch_size: {self._settings.max_batch_size}')
        logger.error(f'max_batch_time: {self._settings.max_batch_time}')
        if GPU:
            self.model  = pipeline(
                task=self.TASK,
                model=self.MODEL_VARIANT,
                device=0, # set device equal to 0 if you want inference on gpu
                batch_size=self._settings.max_batch_size)
        else:
            self.model  = pipeline(
                task=self.TASK,
                model=self.MODEL_VARIANT,
                batch_size=self._settings.max_batch_size)
        logger.error("Loaded on gpu")
        self.loaded = True
        logger.error('model loading complete!')
        return self.loaded

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        if self.loaded == False:
            self.load()
        arrival_time = time.time()
        for request_input in payload.inputs:
            logger.error('request input:\n')
            logger.error(f"{request_input}\n")
            decoded_inputs = self.decode(request_input)
            logger.error('decoded_input:\n')
            logger.error(f"{list(decoded_inputs)}\n")
            X = []
            former_steps_timings = []
            for decoded_input in decoded_inputs:
                json_inputs = json.loads(decoded_input)
                former_steps_timings.append(json_inputs['time'])
                X.append(json_inputs['output']['text'])
        received_batch_len = len(X)
        logger.error(f"recieved batch len:\n{received_batch_len}")
        self.request_counter += received_batch_len
        self.batch_counter += 1
        output = self.model(X)
        serving_time = time.time()
        timing = {
            f"arrival_{PREDICTIVE_UNIT_ID}".replace("-","_"): arrival_time,
            f"serving_{PREDICTIVE_UNIT_ID}".replace("-", "_"): serving_time
        }
        output_with_time = list()
        for pred, former_steps_timing in zip(output, former_steps_timings):
            timing_2_send = deepcopy(timing)
            timing_2_send.update(former_steps_timing)
            print(timing_2_send)
            output_with_time.append(
                {
                    # 'time': timing.update(former_steps_timing),
                    'time': timing_2_send,
                    'output': pred,
                }
            )
        logger.error(f"output_with_time:\n")
        logger.error(output_with_time)
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