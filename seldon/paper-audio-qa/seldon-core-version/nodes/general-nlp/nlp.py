import logging
from textwrap import indent
import torch
from copy import deepcopy
import os
import numpy as np
from transformers import pipeline
from seldon_core.user_model import SeldonResponse
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
            self.MODEL_VARIANT = "distilbert-base-uncased"
            logging.warning(
                f"MODEL_VARIANT env variable not set, using default value: {self.MODEL_VARIANT}")
        try:
            self.TASK = os.environ['TASK']
            logging.info(f'TASK set to: {self.TASK}')
        except KeyError as e:
            self.TASK = 'question-answering'
            logging.warning(
                f"MODEL_VARIANT env variable not set, using default value: {self.TASK}")
        try:
            self.CONTEXT = os.environ['CONTEXT']
            logging.info(f'CONTEXT set to: {self.CONTEXT}')
        except KeyError as e:
            self.CONTEXT = 'default context'
            logging.warning(
                f"CONTEXT env variable not set, using default value: {self.CONTEXT}")

    def load(self):
        logger.info('Loading the ML models')
        self.model  = pipeline(task=self.TASK, model=self.MODEL_VARIANT)
        self.loaded = True
        logger.info('model loading complete!')

    def predict(self, X, features_names=None):
        if self.loaded == False:
            self.load()
        logger.info(f'Incoming input type: {type(X)}')
        logger.info(f"Incoming input:\n{X}\nwas recieved!")
        arrival_time = time.time()
        former_steps_timing = X['time']
        X = X['output']['text']
        qa_input = {
            'question': X,
            'context': self.CONTEXT
        }
        output = self.model(qa_input)
        serving_time = time.time()
        timing = {
            f"arrival_{PREDICTIVE_UNIT_ID}".replace("-","_"): arrival_time,
            f"serving_{PREDICTIVE_UNIT_ID}".replace("-", "_"): serving_time
        }
        timing.update(former_steps_timing)
        output = {
            'time': timing,
            'output': output
        }
        logger.info(f"Output:\n{output}\nwas sent!")
        return output
