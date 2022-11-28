import os
import pathlib
import grpc
from mlserver.grpc.converters import ModelInferResponseConverter
import mlserver.grpc.dataplane_pb2_grpc as dataplane
import mlserver.grpc.converters as converters
from mlserver.codecs.string import StringRequestCodec
import mlserver.types as types
import json
import threading
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)

# single node mlserver
endpoint = "localhost:8081"
model = 'sum-qa'
metadata = []
grpc_channel = grpc.insecure_channel(endpoint)
grpc_stub = dataplane.GRPCInferenceServiceStub(grpc_channel)

# single node seldon+mlserver
# endpoint = "localhost:32000"
# deployment_name = sum-qa
# model = None
# namespace = "default"
# metadata = [("seldon", deployment_name), ("namespace", namespace)]
# grpc_channel = grpc.insecure_channel(endpoint)
# grpc_stub = dataplane.GRPCInferenceServiceStub(grpc_channel)

batch_test = 5

responses = []

PATH = pathlib.Path(__file__).parent.resolve()
with open(os.path.join(PATH, 'input-sample.txt'), 'r') as openfile:
    input_data = openfile.read()
input_data = [input_data]


def send_requests():
    inference_request = types.InferenceRequest(
        inputs=[
            types.RequestInput(
                name="text_inputs",
                shape=[1],
                datatype="BYTES",
                data=[input_data[0].encode('utf8')],
                parameters=types.Parameters(content_type="str"),
            )
        ]
    )
    inference_request_g = converters.ModelInferRequestConverter.from_types(
        inference_request, model_name=model, model_version=None
    )
    response = grpc_stub.ModelInfer(
        request=inference_request_g,
        metadata=metadata)
    responses.append(response)
    return response

thread_pool = []

for i in range(batch_test):
    t = threading.Thread(target=send_requests)
    t.start()
    thread_pool.append(t)

for t in thread_pool:
    t.join()


inference_responses = list(map(
    lambda response: ModelInferResponseConverter.to_types(response), responses))
raw_jsons = list(map(
    lambda inference_response: StringRequestCodec.decode_response(
        inference_response), inference_responses))
outputs = list(map(
    lambda raw_json: json.loads(raw_json[0]), raw_jsons))

pp.pprint(outputs)