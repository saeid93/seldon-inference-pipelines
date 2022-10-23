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
from copy import deepcopy
# import torchvision
import sys
sys.path.insert(0, './cache/ultralytics_yolov5_master')

try:
    PREDICTIVE_UNIT_ID = os.environ['PREDICTIVE_UNIT_ID']
    logger.error(f'PREDICTIVE_UNIT_ID set to: {PREDICTIVE_UNIT_ID}')
except KeyError as e:
    PREDICTIVE_UNIT_ID = 'predictive_unit'
    logger.error(
        f"PREDICTIVE_UNIT_ID env variable not set, using default value: {PREDICTIVE_UNIT_ID}")

class Yolo(MLModel):
    async def load(self):
        self.loaded = False
        self.request_counter = 0
        self.batch_counter = 0
        try:
            self.MODEL_VARIANT = os.environ['MODEL_VARIANT']
            logger.error(f'MODEL_VARIANT set to: {self.MODEL_VARIANT}')
        except KeyError as e:
            self.MODEL_VARIANT = 'yolov5s' 
            logger.error(
                f"MODEL_VARIANT env variable not set, using default value: {self.MODEL_VARIANT}")
        try:
            logger.error('Loading the ML models')
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
            torch.hub.set_dir('./cache')
            logger.error(f'max_batch_size: {self._settings.max_batch_size}')
            logger.error(f'max_batch_time: {self._settings.max_batch_time}')
            self.model = torch.hub.load('ultralytics/yolov5', self.MODEL_VARIANT)
            logger.error('model loaded!')
            self.loaded = True
            logger.error('model loading complete!')
        except OSError:
            raise ValueError('model loading unsuccessful')
        self.loaded = True
        return self.loaded

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        if self.loaded == False:
            self.load()
        arrival_time = time.time()
        for request_input in payload.inputs:
            decoded_input = self.decode(request_input)
            logger.error(f"type of decoded input: {type(decoded_input)}")
            logger.error(f"size of the input: {np.shape(decoded_input)}")
            X = decoded_input.astype(np.uint8)
        if len(X.shape) > 3:
            X = list(X)
        else:
            X = [X]
        logger.error(f'type of X:\n{type(X)}')
        logger.error(f'type of X item:\n{type(X[0])}')
        received_batch_len = len(X)
        logger.error(f"recieved batch len:\n{received_batch_len}")
        self.request_counter += received_batch_len
        self.batch_counter += 1
        objs = self.model(X)
        serving_time = time.time()
        output = self.get_cropped(objs)
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
        str_out = [json.dumps(pred, cls=NumpyEncoder) for pred in output_with_time]
        prediction_encoded = StringCodec.encode_output(payload=str_out, name="output")
        logger.error(f"request counter:\n{self.request_counter}\n")
        logger.error(f"batch counter:\n{self.batch_counter}\n")
        return InferenceResponse(
            id=payload.id,
            model_name=self.name,
            model_version=self.version,
            outputs = [prediction_encoded]
        )

    def get_cropped(self, result):
        """
        crops selected objects for
        the subsequent nodes
        """
        output_list = []
        for res in result.tolist():
            res = res.crop(save=False)
            liscense_labels = ['car', 'truck']
            car_labels = ['car']
            person_labels = ['person']
            res_output = {'person': [], 'car': [], 'liscense': []}
            for obj in res:
                for label in liscense_labels:
                    if label in obj['label']:
                        res_output['liscense'].append(deepcopy(obj['im']))
                        break
                for label in car_labels:
                    if label in obj['label']:
                        res_output['car'].append(deepcopy(obj['im']))
                        break
                for label in person_labels:
                    if label in obj['label']:
                        res_output['person'].append(deepcopy(obj['im']))
                        break
            output_list.append(res_output)
        return output_list
