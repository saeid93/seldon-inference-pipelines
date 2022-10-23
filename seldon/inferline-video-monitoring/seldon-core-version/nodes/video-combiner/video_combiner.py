import logging
import os
import numpy as np
logger = logging.getLogger(__name__)


class VideoCombiner:
    def aggregate(self, features, names=[], meta=[]):
        logger.info(f"features: {features}")
        logger.info(f"features type: {type(features)}")
        logger.info(f"features element: {features[0]}")
        logger.info(f"features element type: {type(features[0])}")
        output = [elm.tolist() for elm in features]
        return output
