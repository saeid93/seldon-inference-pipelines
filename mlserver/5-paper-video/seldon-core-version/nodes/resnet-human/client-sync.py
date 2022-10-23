import os
import requests
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)
import json
from mlserver.types import InferenceResponse
from mlserver.codecs.string import StringRequestCodec
import pathlib

# single node inference
gateway_endpoint="localhost:32000"
deployment_name = 'resnet-human'
namespace = "default"
endpoint = f"http://{gateway_endpoint}/seldon/{namespace}/{deployment_name}/v2/models/infer"

# single node inference
# gateway_endpoint="localhost:8080"
# model='resnet'
# endpoint = f"http://{gateway_endpoint}/v2/models/{model}/infer"

def send_requests(endpoint, data):
    payload = {
        "inputs": [
            {
            "name": "text_inputs",
            "shape": [1],
            "datatype": "BYTES",
            "data": [json.dumps(data)],
            "parameters": {
                "content_type": "str"
            }
            }
        ]
    }
    response = requests.post(endpoint, json=payload)
    return response

PATH = pathlib.Path(__file__).parent.resolve()

with open(os.path.join(PATH, 'input-sample.txt'), 'r') as openfile:
    data = json.load(openfile)

data = [data]

# sync version
results = []
for data_ins in data:
    response = send_requests(endpoint, data_ins)
    results.append(response)

pp.pprint(results[0])
inference_response = InferenceResponse.parse_raw(response.text)
raw_json = StringRequestCodec.decode_response(inference_response)
output = json.loads(raw_json[0])
pp.pprint(output)
