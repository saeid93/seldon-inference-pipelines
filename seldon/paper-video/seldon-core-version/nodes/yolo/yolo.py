import logging
import torch
from copy import deepcopy
import os
import numpy as np
import time
# import torchvision

logger = logging.getLogger(__name__)
PREDICTIVE_UNIT_ID = os.environ['PREDICTIVE_UNIT_ID']

class Yolo(object):
    def __init__(self) -> None:
        super().__init__()
        self.loaded = False
        try:
            self.MODEL_VARIANT = os.environ['MODEL_VARIANT']
            logging.info(f'MODEL_VARIANT set to: {self.MODEL_VARIANT}')
        except KeyError as e:
            self.MODEL_VARIANT = 'yolov5s' 
            logging.warning(
                f"MODEL_VARIANT env variable not set, using default value: {self.MODEL_VARIANT}")

    def load(self):
        try:
            logger.info('Loading the ML models')
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
            self.model = torch.hub.load('ultralytics/yolov5', self.MODEL_VARIANT)
            logger.info('model loaded!')
            # self.resnet.eval()
            self.loaded = True
            logger.info('model loading complete!')
        except OSError:
            pass


    def predict(self, X, features_names=None):
        if self.loaded == False:
            self.load()
        arrival_time = time.time()
        logger.info(f"Incoming input:\n{X}\nwas recieved!")
        logger.info(f"input type: {type(X)}")
        logger.info(f"input shape: {X.shape}") 
        X = np.array(X, dtype=np.uint8)
        logger.info(f"output type: {type(X)}")
        logger.info(f"output shape: {X.shape}")
        objs = self.model(X)
        serving_time = time.time()
        output = self.get_cropped(objs)
        logger.info(f"arrival time {PREDICTIVE_UNIT_ID}: {arrival_time}")
        logger.info(f"serving time {PREDICTIVE_UNIT_ID}: {serving_time}")
        # output[f'arrival_{PREDICTIVE_UNIT_ID}'] = arrival_time
        # output[f'serving_{PREDICTIVE_UNIT_ID}'] = serving_time
        output['time'] = {
            f'arrival_{PREDICTIVE_UNIT_ID}': arrival_time,
            f'serving_{PREDICTIVE_UNIT_ID}': serving_time
        }
        logger.info(f"Output:\n{output}\nwas sent!")
        return output

    def get_cropped(self, result):
        """
        crops selected objects for
        the subsequent nodes
        """
        result = result.crop()
        liscense_labels = ['car', 'truck']
        car_labels = ['car']
        person_labels = ['person']
        output_list = {'person': [], 'car': [], 'liscense': []}
        for obj in result:
            for label in liscense_labels:
                if label in obj['label']:
                    output_list['liscense'].append(deepcopy(obj['im'].tolist()))
                    break
            for label in car_labels:
                if label in obj['label']:
                    output_list['car'].append(deepcopy(obj['im'].tolist()))
                    break
            for label in person_labels:
                if label in obj['label']:
                    output_list['person'].append(deepcopy(obj['im'].tolist()))
                    break
        return output_list
