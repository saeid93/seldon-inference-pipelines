import logging
import os
import numpy as np
from torchvision import transforms
from PIL import Image
logger = logging.getLogger(__name__)


class Transformer:
    def __init__(self) -> None:
        super().__init__()
        # standard resnet image transformation
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )])

    def transform_input(self, X, names=[], meta=[]):
        logger.info(f"Transforming input: {X}")
        X_trans = Image.fromarray(X.astype(np.uint8))
        X_trans = self.transform(X_trans)
        X_trans = np.array(X_trans).tolist()
        return X_trans
