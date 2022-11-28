from urllib import response
import requests
from pprint import PrettyPrinter
from mlserver.types import InferenceResponse
from mlserver.codecs.string import StringRequestCodec
pp = PrettyPrinter(indent=4)
from datasets import load_dataset
import threading
import json

# single node inferline
gateway_endpoint = "localhost:32000"
deployment_name = 'mock-one'
namespace = "default"
endpoint = f"http://{gateway_endpoint}/seldon/{namespace}/{deployment_name}/v2/models/infer"

# single node inferline
# gateway_endpoint = "localhost:8080"
# model = 'mock-one'
# endpoint = f"http://{gateway_endpoint}/v2/models/{model}/infer"

batch_test = 30
responses = []

ds = load_dataset(
    "hf-internal-testing/librispeech_asr_demo",
    "clean",
    split="validation")

input_data = ds[0]["audio"]["array"]

def send_requests():
    payload = {
        "inputs": [
            {
            "name": "array_inputs",
            "shape": [1, len(input_data)],
            "datatype": "FP32",
            "data": input_data.tolist(),
            "parameters": {
                "content_type": "np"
            }
            }
        ]
    }
    response = requests.post(endpoint, json=payload)
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
    lambda response: InferenceResponse.parse_raw(response.text), responses))
raw_jsons = list(map(
    lambda inference_response: StringRequestCodec.decode_response(
        inference_response), inference_responses))
outputs = list(map(
    lambda raw_json: json.loads(raw_json[0]), raw_jsons))

pp.pprint(outputs)