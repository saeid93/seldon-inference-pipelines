from urllib import response
import grpc
from pprint import PrettyPrinter
from mlserver.types import InferenceResponse
from mlserver.grpc.converters import ModelInferResponseConverter
import mlserver.grpc.dataplane_pb2_grpc as dataplane
import mlserver.grpc.converters as converters
from mlserver.codecs.string import StringRequestCodec
from mlserver.codecs.string import StringRequestCodec
pp = PrettyPrinter(indent=4)
from datasets import load_dataset
import mlserver.types as types
import json
import asyncio

# single node mlserver
endpoint = "localhost:8081"
model = 'audio'
metadata = []

# single node seldon+mlserver
# endpoint = "localhost:32000"
# deployment_name = 'audio'
# model = 'audio'
# namespace = "default"
# metadata = [("seldon", deployment_name), ("namespace", namespace)]

batch_test = 5
ds = load_dataset(
    "hf-internal-testing/librispeech_asr_demo",
    "clean",
    split="validation")

input_data = ds[0]["audio"]["array"]

async def send_requests(ch):
    grpc_stub = dataplane.GRPCInferenceServiceStub(ch)
    inference_request = types.InferenceRequest(
        inputs=[
            types.RequestInput(
                name="echo_request",
                shape=[1, len(input_data)],
                datatype="FP32",
                data=input_data.tolist(),
                parameters=types.Parameters(content_type="np"),
            )
        ]
    )
    inference_request_g = converters.ModelInferRequestConverter.from_types(
        inference_request, model_name=model, model_version=None
    )
    response = await grpc_stub.ModelInfer(
        request=inference_request_g,
        metadata=metadata)
    return response


async def main():
    async with grpc.aio.insecure_channel(endpoint) as ch:
        responses = await asyncio.gather(*[send_requests(ch) for _ in range(10)])

    inference_responses = list(map(
        lambda response: ModelInferResponseConverter.to_types(response), responses))
    raw_jsons = list(map(
        lambda inference_response: StringRequestCodec.decode_response(
            inference_response), inference_responses))
    outputs = list(map(
        lambda raw_json: json.loads(raw_json[0]), raw_jsons))

    pp.pprint(outputs)

asyncio.run(main())