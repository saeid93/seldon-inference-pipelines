import os
import time
import json
import asyncio
import psutil
from mlserver import MLModel
import numpy as np
from mlserver.logging import logger
from mlserver.utils import get_model_uri
from mlserver.types import (
    InferenceRequest,
    InferenceResponse,
    ResponseOutput,
    Parameters)
from mlserver import MLModel
from mlserver.codecs import StringCodec
from mlserver_huggingface.common import NumpyEncoder
from typing import List, Dict
import time


try:
    PREDICTIVE_UNIT_ID = os.environ['PREDICTIVE_UNIT_ID']
    logger.error(f'PREDICTIVE_UNIT_ID set to: {PREDICTIVE_UNIT_ID}')
except KeyError as e:
    PREDICTIVE_UNIT_ID = 'mock_one'
    logger.error(
        f"PREDICTIVE_UNIT_ID env variable not set, using default value: {PREDICTIVE_UNIT_ID}")

async def model(input, sleep):
    await asyncio.sleep(sleep)
    _ = input
    output = ["mock one output"] * len(input)
    return output
    
class MockOne(MLModel):
    async def load(self):
        self.loaded = False
        self.request_counter = 0
        self.batch_counter = 0
        try:
            self.MODEL_VARIANT = float(os.environ['MODEL_VARIANT'])
            logger.error(f'MODEL_VARIANT set to: {self.MODEL_VARIANT}')
        except KeyError as e:
            self.MODEL_VARIANT = 0
            logger.error(
                f"MODEL_VARIANT env variable not set, using default value: {self.MODEL_VARIANT}")
        logger.info('Loading the ML models')
        # TODO add batching like the runtime
        logger.error(f'max_batch_size: {self._settings.max_batch_size}')
        logger.error(f'max_batch_time: {self._settings.max_batch_time}')
        self.model  = model
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
        output: List[Dict] = await self.model(X, self.MODEL_VARIANT)
        logger.error(f"model output:\n{output}")
        serving_time = time.time()
        timing = {
            f"arrival_{PREDICTIVE_UNIT_ID}".replace("-","_"): arrival_time,
            f"serving_{PREDICTIVE_UNIT_ID}".replace("-", "_"): serving_time
        }
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
