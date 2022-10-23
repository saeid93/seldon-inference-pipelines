import logging
import random
import os
from numpy import indices

# NUMBER_OF_ROUTES = int(os.environ.get("NUMBER_OF_ROUTES", "2"))


class Router:
    def __init__(self) -> None:
        super().__init__()
        logging.info("Router initialized!")

    def route(self, features, names=[], meta=[]):
        logging.info(f"Router received features:\n{features}\nand names:\n{names}\nand meta:\n{meta}")
        route = features["route"]
        return route