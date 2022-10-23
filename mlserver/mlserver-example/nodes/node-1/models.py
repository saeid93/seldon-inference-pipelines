from itertools import count
from mlserver import MLModel
import numpy as np
from mlserver.codecs import NumpyCodec
import json
from mlserver.logging import logger
from mlserver.utils import get_model_uri
from mlserver.types import InferenceRequest, InferenceResponse, ResponseOutput
from mlserver import MLModel
from mlserver.codecs import DecodedParameterName
from mlserver.cli.serve import load_settings
# import logging

# logger = logging.getLogger("code.simple")

_to_exclude = {
    "parameters": {DecodedParameterName, "headers"},
    'inputs': {"__all__": {"parameters": {DecodedParameterName, "headers"}}}
}

def model(input):
  output =  input * 2
  return output

class NodeOne(MLModel):
  async def load(self) -> bool:
    self._model = model
    self.ready = True
    self.counter = 0
    logger.error("This is from the logger")
    return self.ready

  async def predict(self, payload: InferenceRequest) -> InferenceResponse:
      outputs = []
      self.counter += 1
      logger.error(f"counter: {self.counter}")
      logger.error('*'*50)
      logger.error("This is recieved request")
      logger.error(payload.inputs)
      for request_input in payload.inputs:
          decoded_input = self.decode(request_input)
          logger.error(f"decoded_input:\n{decoded_input}")
          logger.error(f"size of the input: {np.shape(decoded_input)}")
          model_output = self._model(decoded_input)
          outputs.append(
              ResponseOutput(
                  name=request_input.name,
                  datatype=request_input.datatype,
                  parameters={
                    "content_type": "np"
                  },
                  shape=request_input.shape,
                  data=model_output.tolist()
              )
          )

      return InferenceResponse(model_name=self.name, outputs=outputs)
