from pprint import PrettyPrinter
from mlserver.grpc.converters import ModelInferResponseConverter
import mlserver.grpc.dataplane_pb2_grpc as dataplane
import mlserver.grpc.converters as converters
from mlserver.codecs.string import StringRequestCodec
from datasets import load_dataset
import mlserver.types as types
import json
import grpc

pp = PrettyPrinter(indent=4)

# single node mlserver
endpoint = "localhost:8081"
model = 'audio'
metadata = []
grpc_channel = grpc.insecure_channel(endpoint)
grpc_stub = dataplane.GRPCInferenceServiceStub(grpc_channel)

# single node seldon+mlserver
# endpoint = "localhost:32000"
# deployment_name = 'audio'
# model = 'audio'
# namespace = "default"
# metadata = [("seldon", deployment_name), ("namespace", namespace)]
# grpc_channel = grpc.insecure_channel(endpoint)
# grpc_stub = dataplane.GRPCInferenceServiceStub(grpc_channel)


batch_test = 6

ds = load_dataset(
    "hf-internal-testing/librispeech_asr_demo",
    "clean",
    split="validation")

input_data = ds[0]["audio"]["array"]

def send_requests():
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
    response = grpc_stub.ModelInfer(
        request=inference_request_g,
        metadata=metadata)

    return response

thread_pool = []

# sync version
results = []
for i in range(batch_test):
    response = send_requests()
    results.append(response)


# Note that here we just convert from the gRPC types to the MLServer types
inference_response = ModelInferResponseConverter.to_types(response)
raw_json = StringRequestCodec.decode_response(inference_response)
output = json.loads(raw_json[0])
pp.pprint(output)