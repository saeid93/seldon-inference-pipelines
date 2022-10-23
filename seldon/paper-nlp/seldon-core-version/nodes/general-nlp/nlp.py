import logging
from textwrap import indent
import torch
from copy import deepcopy
import os
import numpy as np
from transformers import pipeline
from pprint import PrettyPrinter
import time

logger = logging.getLogger(__name__)
PREDICTIVE_UNIT_ID = os.environ['PREDICTIVE_UNIT_ID']

class GeneralNLP(object):
    def __init__(self) -> None:
        super().__init__()
        self.loaded = False
        try:
            self.MODEL_VARIANT = os.environ['MODEL_VARIANT']
            logging.info(f'MODEL_VARIANT set to: {self.MODEL_VARIANT}')
        except KeyError as e:
            self.MODEL_VARIANT = 'sshleifer/distilbart-cnn-12-6'
            logging.warning(
                f"MODEL_VARIANT env variable not set, using default value: {self.MODEL_VARIANT}")
        try:
            self.TASK = os.environ['TASK']
            logging.info(f'TASK set to: {self.TASK}')
        except KeyError as e:
            self.TASK = 'summarization' 
            logging.warning(
                f"MODEL_VARIANT env variable not set, using default value: {self.TASK}")

    def load(self):
        logger.info('Loading the ML models')
        self.model  = pipeline(task=self.TASK, model=self.MODEL_VARIANT)
        self.loaded = True
        logger.info('model loading complete!')

    def predict(self, X, features_names=None):
        if self.loaded == False:
            self.load()
        former_steps_timing = None
        arrival_time = time.time()
        logger.info(f'Incoming input type: {type(X)}')
        logger.info(f"Incoming input:\n{X}\nwas recieved!")
        if type(X) is not str: # If not the first node
            former_steps_timing = X['time']
            if type(X) is dict and 'label' in X.keys():    
                label = X['label'] # To be later used for node logic
                X = X['input']
            else:
                logger.info("Here!!")
                X = X['output']
                X = list(X.values())[0]
        output = self.model(X)
        logger.info(f"model output:\n{output}")
        serving_time = time.time()
        timing = {
            f"arrival_{PREDICTIVE_UNIT_ID}".replace("-","_"): arrival_time,
            f"serving_{PREDICTIVE_UNIT_ID}".replace("-", "_"): serving_time
        }
        if former_steps_timing is not None:
            timing.update(former_steps_timing)
        if self.TASK == "text-classification":
            output = {
                'time': timing,
                'label': output[0]['label'],
                'input': X}
        else:
            output = {
                'time': timing,
                'output': output[0],                
            }
        logger.info(f"Output:\n{output}\nwas sent!")
        return output
