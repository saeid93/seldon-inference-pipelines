from urllib import response
import requests
from pprint import PrettyPrinter
from mlserver.types import InferenceResponse
from mlserver.codecs.string import StringRequestCodec
pp = PrettyPrinter(indent=4)
from transformers import pipeline
from datasets import load_dataset
import threading
import json

# single node inferline
gateway_endpoint = "localhost:32000"
deployment_name = 'mlserver-mock'
namespace = "default"
endpoint = f"http://{gateway_endpoint}/seldon/{namespace}/{deployment_name}/v2/models/infer"

batch_test = 1

ds = load_dataset(
    "hf-internal-testing/librispeech_asr_demo",
    "clean",
    split="validation")

input_data = ds[0]["audio"]["array"]

def send_requests(endpoint):
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
    return response

# sync version
results = []
for i in range(batch_test):
    response = send_requests(endpoint)
    results.append(response)

pp.pprint(results[0])
inference_response = InferenceResponse.parse_raw(response.text)
raw_json = StringRequestCodec.decode_response(inference_response)
output = json.loads(raw_json[0])
pp.pprint(output)
