import os
import time
from copy import deepcopy
from mlserver import MLModel
import numpy as np
from mlserver.codecs import NumpyCodec
import json
from mlserver.logging import logger
from mlserver.utils import get_model_uri
from mlserver.types import (
    InferenceRequest,
    InferenceResponse,
    ResponseOutput,
    Parameters)
from mlserver.codecs.string import StringRequestCodec
from mlserver import MLModel
from mlserver.codecs import DecodedParameterName
from mlserver.cli.serve import load_settings
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
            self.MODEL_VARIANT = "distilbert-base-uncased"
            logger.error(
                f"MODEL_VARIANT env variable not set, using default value: {self.MODEL_VARIANT}")
        try:
            self.TASK = os.environ['TASK']
            logger.error(f'TASK set to: {self.TASK}')
        except KeyError as e:
            self.TASK = 'question-answering'
            logger.error(
                f"MODEL_VARIANT env variable not set, using default value: {self.TASK}")
        try:
            self.CONTEXT = os.environ['CONTEXT']
            logger.error(f'CONTEXT set to: {self.CONTEXT}')
        except KeyError as e:
            self.CONTEXT = 'default context'
            logger.error(
                f"CONTEXT env variable not set, using default value: {self.CONTEXT}")
        logger.error('Loading the ML models')
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
                X.append({
                    'question': json_inputs['output']['text'],
                    'context': self.CONTEXT                    
                })
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
        preds_with_time = list()
        for pred, former_steps_timing in zip(output, former_steps_timings):
            timing_2_send = deepcopy(timing)
            timing_2_send.update(former_steps_timing)
            print(timing_2_send)
            preds_with_time.append(
                {
                    # 'time': timing.update(former_steps_timing),
                    'time': timing_2_send,
                    'output': pred,
                }
            )
        logger.error(f"output_with_time:\n")
        logger.error(preds_with_time)
        str_out = [
            json.dumps(pred, cls=NumpyEncoder) for pred in preds_with_time]
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
