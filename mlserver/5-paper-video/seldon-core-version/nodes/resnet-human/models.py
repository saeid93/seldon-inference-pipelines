import os
import torch
from torchvision import models
from torchvision import transforms
from PIL import Image
import time
from mlserver import MLModel
import numpy as np
from mlserver.codecs import NumpyCodec
from mlserver.logging import logger
from mlserver.utils import get_model_uri
from mlserver.types import (
    InferenceRequest,
    InferenceResponse,
    ResponseOutput)
from mlserver import MLModel
from mlserver.codecs import DecodedParameterName
from mlserver.cli.serve import load_settings
from mlserver.codecs import StringCodec
from mlserver_huggingface.common import NumpyEncoder
from copy import deepcopy
import json

try:
    PREDICTIVE_UNIT_ID = os.environ['PREDICTIVE_UNIT_ID']
    logger.error(f'PREDICTIVE_UNIT_ID set to: {PREDICTIVE_UNIT_ID}')
except KeyError as e:
    PREDICTIVE_UNIT_ID = 'resnet'
    logger.error(
        f"PREDICTIVE_UNIT_ID env variable not set, using default value: {PREDICTIVE_UNIT_ID}")


# PREDICTIVE_UNIT_ID = 'name' #os.environ['PREDICTIVE_UNIT_ID']

class ResnetHuman(MLModel):
    async def load(self) -> bool:
        self.loaded = False
        self.request_counter = 0
        self.batch_counter = 0
        # standard resnet image transformation
        try:
            self.MODEL_VARIANT = os.environ['MODEL_VARIANT']
            logger.error(f'MODEL_VARIANT set to: {self.MODEL_VARIANT}')
        except KeyError as e:
            self.MODEL_VARIANT = 'resnet101'
            logger.error(
                f"MODEL_VARIANT env variable not set, using default value: {self.MODEL_VARIANT}")
        logger.error(f'max_batch_size: {self._settings.max_batch_size}')
        logger.error(f'max_batch_time: {self._settings.max_batch_time}')
        self.batch_size = self._settings.max_batch_size
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])
        logger.error('Init function complete!')
        model = {
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
            'resnet101': models.resnet101,
            'resnet152': models.resnet152,
        }
        logger.error('Loading the ML models')
        # TODO cpu and gpu from env variable
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.resnet = model[self.MODEL_VARIANT](pretrained=True)
        self.resnet.eval()
        self.loaded = True
        logger.error('model loading complete!')
        return self.loaded

    async def predict(self, payload: InferenceRequest) -> InferenceRequest:
        if self.loaded == False:
            self.load()
        arrival_time = time.time()
        for request_input in payload.inputs:
            logger.error('request input:\n')
            decoded_inputs = self.decode(request_input)
            X = []
            former_steps_timings = []
            for decoded_input in decoded_inputs:
                json_inputs = json.loads(decoded_input)
                former_steps_timings.append(json_inputs['time'])
                X.append(json_inputs['output']['person'])
        # TODO add handling of zero person in the iamge
        if X == []:
            return []
        received_batch_len = len(X)
        self.request_counter += received_batch_len
        self.batch_counter += 1

        # First all images recieved in the last step
        mask = []
        X_flatten = []
        for output_index, output in enumerate(X):
            mask.append(0)
            for image in output:
                mask[output_index] += 1
                X_flatten.append(image)

        logger.error(f"len(X_flatten):\n{len(X_flatten)}\n")
        logger.error(f"mask:\n{mask}\n")
        X_trans = [
            self.transform(
                Image.fromarray(
                    np.array(image).astype(np.uint8))) for image in X_flatten]

        # give all the images to the models per defined batch size
        predictions = []
        complete_batch = torch.stack(X_trans, axis=0)
        timings = []
        for i in range(0, len(complete_batch), self.batch_size):
            out = self.resnet(complete_batch[i: i + self.batch_size])
            percentages = torch.nn.functional.softmax(out, dim=1) * 100
            percentages = percentages.detach().numpy()
            image_net_class = np.argmax(percentages, axis=1)
            predictions += image_net_class.tolist()
            logger.error(f"{image_net_class=}")
            serving_time = time.time()
            timing = {
                f"arrival_{PREDICTIVE_UNIT_ID}": arrival_time,
                f"serving_{PREDICTIVE_UNIT_ID}": serving_time,
            }
            timings += self.batch_size * [timing]
        logger.error(f"{timings=}")
        logger.error(f"{former_steps_timings=}")
        timing.update(former_steps_timings)
        preds_with_time = list()
        pred_index = 0
        for pred_mask, former_step_timing in zip(mask, former_steps_timings):
            this_node_times = []
            this_input_predictions = []
            while pred_mask != 0:
                this_node_times.append(timings[pred_index])
                this_input_predictions.append(predictions[pred_index])
                pred_index += 1
                pred_mask -= 1
                logger.error(f"{pred_index=}")
                logger.error(f"{pred_mask=}")
            former_step_timing[f"{PREDICTIVE_UNIT_ID}_times"] = this_node_times
            preds_with_time.append({
                f"time": former_step_timing,
                'output': this_input_predictions,
                })
        logger.error(f"output_with_time:\n")
        logger.error(preds_with_time)
        str_out = [json.dumps(
            pred, cls=NumpyEncoder) for pred in preds_with_time]
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
