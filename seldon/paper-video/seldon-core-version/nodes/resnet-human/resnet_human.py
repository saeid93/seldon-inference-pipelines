import os
import logging
import numpy as np
import torch
from torchvision import models
from torchvision import transforms
from PIL import Image
import time

logger = logging.getLogger(__name__)
PREDICTIVE_UNIT_ID = os.environ['PREDICTIVE_UNIT_ID']

class ResnetHuman(object):
    def __init__(self) -> None:
        super().__init__()
        self.loaded = False
        # standard resnet image transformation
        try:
            self.MODEL_VARIANT = os.environ['MODEL_VARIANT']
            logging.info(f'MODEL_VARIANT set to: {self.MODEL_VARIANT}')
        except KeyError as e:
            self.MODEL_VARIANT = 'resnet101' 
            logging.warning(
                f"MODEL_VARIANT env variable not set, using default value: {self.MODEL_VARIANT}")
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])
        logger.info('Init function complete!')

    def load(self):
        model = {
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
            'resnet101': models.resnet101,
            'resnet152': models.resnet152,
        }
        logger.info('Loading the ML models')
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.resnet = model[self.MODEL_VARIANT](pretrained=True)
        self.resnet.eval()
        self.loaded = True
        logger.info('model loading complete!')

    def predict(self, X, features_names=None):
        if self.loaded == False:
            self.load()
        # logger.info(f"Incoming input:\n{X}\nwas recieved!")
        arrival_time = time.time()
        former_steps_timing = X['time']
        # arrival_yolo = X['time']['arrival_yolo']
        # serving_yolo = X['time']['serving_yolo']
        # logger.info(f"arrival time yolo: {arrival_yolo}")
        # logger.info(f"serving time yolo: {serving_yolo}")
        if X['person'] == []:
            return []
        X = X['person']
        # if self.WITH_PREPROCESSOR:
        #     X_trans = torch.from_numpy(X.astype(np.float32))
        # else:
        #     # X_trans = Image.fromarray(X.astype(np.uint8))
        #     # X_trans = self.transform(X_trans)
        X_trans = [
            self.transform(
                Image.fromarray(
                    np.array(image).astype(np.uint8))) for image in X]
        batch = torch.stack(X_trans, axis=0)
        out = self.resnet(batch)
        percentages = torch.nn.functional.softmax(out, dim=1) * 100
        percentages = percentages.detach().numpy()
        image_net_class = np.argmax(percentages, axis=1)
        serving_time = time.time()
        # logger.info(f"arrival time yolo: {arrival_resnet}")
        # logger.info(f"serving time yolo: {serving_resnet}")
        timing = {
            f"arrival_{PREDICTIVE_UNIT_ID}": arrival_time,
            f"serving_{PREDICTIVE_UNIT_ID}": serving_time,
        }
        timing.update(former_steps_timing)
        output = {
            "time": timing,
            "output": image_net_class.tolist()
        }
        logger.info(f"Output:\n{output}\nwas sent!")
        return output
