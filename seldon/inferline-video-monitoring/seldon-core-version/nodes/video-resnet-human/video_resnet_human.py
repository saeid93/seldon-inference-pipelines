import os
import logging
import numpy as np
import torch
from torchvision import models
from torchvision import transforms
from PIL import Image

logger = logging.getLogger(__name__)

class VideoResnetHuman(object):
    def __init__(self) -> None:
        super().__init__()
        self.loaded = False
        # standard resnet image transformation
        # try:
        #     self.WITH_PREPROCESSOR = bool(os.environ['WITH_PREPROCESSOR'])
        #     logging.info(f'WITH_PREPROCESSOR set to: {self.WITH_PREPROCESSOR}')
        # except KeyError as e:
        #     self.WITH_PREPROCESSOR = False
        #     logging.warning(
        #         f"WITH_PREPROCESSOR env variable not set, using default value: {self.WITH_PREPROCESSOR}")
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
        logger.info('Loading the ML models')
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.resnet = models.resnet101(pretrained=True)
        self.resnet.eval()
        self.loaded = True
        logger.info('model loading complete!')

    def predict(self, X, features_names=None):
        if self.loaded == False:
            self.load()
        logger.info(f"Incoming input:\n{X}\nwas recieved!")
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
        return image_net_class.tolist()
