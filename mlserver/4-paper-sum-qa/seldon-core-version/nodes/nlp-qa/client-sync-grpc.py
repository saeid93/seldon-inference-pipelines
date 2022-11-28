import os
import pathlib
import grpc
from mlserver.grpc.converters import ModelInferResponseConverter
import mlserver.grpc.dataplane_pb2_grpc as dataplane
import mlserver.grpc.converters as converters
from mlserver.codecs.string import StringRequestCodec
import mlserver.types as types
import json
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)

# single node mlserver
endpoint = "localhost:8081"
model = 'nlp-qa'
metadata = []
grpc_channel = grpc.insecure_channel(endpoint)
grpc_stub = dataplane.GRPCInferenceServiceStub(grpc_channel)

# single node seldon+mlserver
# endpoint = "localhost:32000"
# deployment_name = 'nlp-qa'
# model = 'nlp-qa'
# namespace = "default"
# metadata = [("seldon", deployment_name), ("namespace", namespace)]
# grpc_channel = grpc.insecure_channel(endpoint)
# grpc_stub = dataplane.GRPCInferenceServiceStub(grpc_channel)

PATH = pathlib.Path(__file__).parent.resolve()

with open(os.path.join(PATH, 'input-sample.txt'), 'r') as openfile:
    input_data = openfile.read()

input_data = [input_data]


def send_requests(input_data):
    inference_request = types.InferenceRequest(
        inputs=[
            types.RequestInput(
                name="text_inputs",
                shape=[1],
                datatype="BYTES",
                data=[input_data.encode('utf8')],
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
    return response


# sync version
results = []
for data_ins in input_data:
    response = send_requests(data_ins)
    results.append(response)

# Note that here we just convert from the gRPC types to the MLServer types
inference_response = ModelInferResponseConverter.to_types(response)
raw_json = StringRequestCodec.decode_response(inference_response)
output = json.loads(raw_json[0])
pp.pprint(output)
