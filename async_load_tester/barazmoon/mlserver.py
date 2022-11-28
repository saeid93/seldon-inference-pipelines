from typing import List, Tuple, Any

import mlserver.grpc.converters as converters
import aiohttp.payload as aiohttp_payload
import mlserver.types as types

from .main import BarAzmoonProcess
from .main import BarAzmoonAsyncRest
from .main import BarAzmoonAsyncGrpc


class MLServerProcess(BarAzmoonProcess):
    def __init__(
        self, *, workload: List[int],
        endpoint: str, http_method="get", **kwargs):
        super().__init__(
            workload=workload, endpoint=endpoint,
            http_method=http_method, **kwargs)
        self.data_type = self.kwargs['data_type']

    def get_request_data(self) -> Tuple[str, str]:
        if self.data_type == 'example':
            payload = {
                "inputs": [
                    {
                        "name": "parameters-np",
                        "datatype": "FP32",
                        "shape": self.kwargs['data_shape'],
                        "data": self.kwargs['data'],
                        "parameters": {
                            "content_type": "np"
                        }
                    }]
                }
        elif self.data_type == 'audio':
            payload = {
                "inputs": [
                    {
                    "name": "array_inputs",
                    "shape": self.kwargs['data_shape'],
                    "datatype": "FP32",
                    "data": self.kwargs['data'],
                    "parameters": {
                        "content_type": "np"
                    }
                    }
                ]
            }
        elif self.data_type == 'text':
            payload = {
                "inputs": [
                    {
                        "name": "text_inputs",
                        "shape": self.kwargs['data_shape'],
                        "datatype": "BYTES",
                        "data": self.kwargs['data'],
                        "parameters": {
                            "content_type": "str"
                        }
                    }
                ]
            }
        elif self.data_type == 'image':
            payload = {
                "inputs":[
                    {
                        "name": "parameters-np",
                        "datatype": "INT32",
                        "shape": self.kwargs['data_shape'],
                        "data": self.kwargs['data'],
                        "parameters": {
                            "content_type": "np"
                            }
                    }]
                }
        else:
            raise ValueError(f"Unkown datatype {self.kwargs['data_type']}")
        return None, aiohttp_payload.JsonPayload(payload)

    def process_response(self, data_id: str, response: dict):
        if self.data_type == 'image':
            print(f"{data_id}=")
            # print(f"{response.keys()=}")
        else:
            print(f"{data_id}=")
            print(response)


class MLServerAsyncRest:
    def __init__(
        self, *, workload: List[int], endpoint: str,
        data: Any, data_shape: List[int],
        data_type: str, http_method = "post",
        **kwargs,):
        self.endpoint = endpoint
        self.http_method = http_method
        self._workload = (rate for rate in workload)
        self._counter = 0
        self.data_type = data_type
        self.data = data
        self.data_shape = data_shape
        self.kwargs = kwargs
        _, self.payload = self.get_request_data()

    async def start(self):
        c = BarAzmoonAsyncRest(self.endpoint, self.payload)
        await c.benchmark(self._workload)
        await c.close()
        return c.responses

    def get_request_data(self) -> Tuple[str, str]:
        if self.data_type == 'example':
            payload = {
                "inputs": [
                    {
                        "name": "parameters-np",
                        "datatype": "FP32",
                        "shape": self.data_shape,
                        "data": self.data,
                        "parameters": {
                            "content_type": "np"
                        }
                    }]
                }
        elif self.data_type == 'audio':
            payload = {
                "inputs": [
                    {
                    "name": "array_inputs",
                    "shape": self.data_shape,
                    "datatype": "FP32",
                    "data": self.data,
                    "parameters": {
                        "content_type": "np"
                    }
                    }
                ]
            }
        elif self.data_type == 'text':
            payload = {
                "inputs": [
                    {
                        "name": "text_inputs",
                        "shape": self.data_shape,
                        "datatype": "BYTES",
                        "data": [self.data],
                        "parameters": {
                            "content_type": "str"
                        }
                    }
                ]
            }
        elif self.data_type == 'image':
            payload = {
                "inputs":[
                    {
                        "name": "parameters-np",
                        "datatype": "INT32",
                        "shape": self.data_shape,
                        "data": self.data,
                        "parameters": {
                            "content_type": "np"
                            }
                    }]
                }
        else:
            raise ValueError(f"Unkown datatype {self.kwargs['data_type']}")
        return None, aiohttp_payload.JsonPayload(payload)


class MLServerAsyncGrpc:
    # TODO
    def __init__(
        self, *, workload: List[int], endpoint: str,
        data: Any, data_shape: List[int], model: str,
        data_type: str, metadata: List[Tuple[str, str]],
        **kwargs,):
        self.endpoint = endpoint
        self.metadata = metadata
        self.model = model
        self._workload = (rate for rate in workload)
        self._counter = 0
        self.data_type = data_type
        self.data = data
        self.data_shape = data_shape
        self.kwargs = kwargs
        _, self.payload = self.get_request_data()

    async def start(self):
        c = BarAzmoonAsyncGrpc(self.endpoint, self.metadata, self.payload)
        await c.benchmark(self._workload)
        return c.responses

    def get_request_data(self) -> Tuple[str, str]:
        if self.data_type == 'audio':
            payload = types.InferenceRequest(
                inputs=[
                    types.RequestInput(
                        name="echo_request",
                        shape=self.data_shape,
                        datatype="FP32",
                        data=self.data,
                        parameters=types.Parameters(content_type="np"),
                        )
                    ]
                )
        elif self.data_type == 'text':
            payload = types.InferenceRequest(
                inputs=[
                    types.RequestInput(
                        name="text_inputs",
                        shape=[1],
                        datatype="BYTES",
                        data=[self.data.encode('utf8')],
                        parameters=types.Parameters(content_type="str"),
                        )
                    ]
                )
        elif self.data_type == 'image':
            payload =  types.InferenceRequest(
                inputs=[
                    types.RequestInput(
                    name="parameters-np",
                    shape=self.data_shape,
                    datatype="INT32",
                    data=self.data,
                    parameters=types.Parameters(content_type="np"),
                    )
                ]
            )
        else:
            raise ValueError(f"Unkown datatype {self.kwargs['data_type']}")
        payload = converters.ModelInferRequestConverter.from_types(
            payload, model_name=self.model, model_version=None
        )
        return None, payload
