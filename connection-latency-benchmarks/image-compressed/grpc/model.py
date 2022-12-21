import os
import torch
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
from mlserver.codecs import StringCodec
from mlserver_huggingface.common import NumpyEncoder
from typing import List, Dict
from mlserver import types
import base64
import asyncio

try:
    PREDICTIVE_UNIT_ID = os.environ['PREDICTIVE_UNIT_ID']
    logger.error(f'PREDICTIVE_UNIT_ID set to: {PREDICTIVE_UNIT_ID}')
except KeyError as e:
    PREDICTIVE_UNIT_ID = 'mock-one'
    logger.error(
        f"PREDICTIVE_UNIT_ID env variable not set, using default value: {PREDICTIVE_UNIT_ID}")

try:
    MODEL_SYNC = eval(os.environ['MODEL_SYNC'])
    logger.info(f'MODEL_SYNC set to: {MODEL_SYNC}')
except KeyError as e:
    MODEL_SYNC = True
    logger.error(
        f"PREDICTIVE_UNIT_ID env variable not set, using default value: {MODEL_SYNC}")

def decode_from_bin(inp, shape, dtype):
    buff = memoryview(inp)
    im_array = np.frombuffer(buff, dtype=dtype).reshape(shape)
    return im_array

async def model(input, sleep):
    if MODEL_SYNC:
        time.sleep(sleep)
    else:
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
            self.MODEL_VARIANT = 0.3
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
        arrival_time = time.time()
        for request_input in payload.inputs:
            dtype = request_input.parameters.dtype
            shape = eval(request_input.parameters.datashape)
            data = request_input.data.__root__
            X = decode_from_bin(inp=data[0], shape=shape, dtype=dtype)
        if len(X.shape) > 3:
            X = list(X)       
        else:
            X = [X]
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
        times = {
            PREDICTIVE_UNIT_ID: {
            "arrival": arrival_time,
            "serving": serving_time
            }
        }
        str_out = [json.dumps(
            pred, cls=NumpyEncoder) for pred in output]
        prediction_encoded = StringCodec.encode_output(
            payload=str_out, name="output")
        logger.error(f"Output:\n{prediction_encoded}\nwas sent!")
        logger.error(f"request counter:\n{self.request_counter}\n")
        logger.error(f"batch counter:\n{self.batch_counter}\n")
        return InferenceResponse(
            id=payload.id,
            model_name=self.name,
            model_version=self.version,
            outputs = [prediction_encoded],
            parameters=types.Parameters(times=str(times))
        )
