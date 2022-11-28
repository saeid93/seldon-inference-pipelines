import os
import json
import requests
import threading
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)
from mlserver.types import InferenceResponse
from mlserver.codecs.string import StringRequestCodec
from pprint import PrettyPrinter
import pathlib
import numpy as np

pp = PrettyPrinter(indent=1)

# single node inference
model = 'resnet'
gateway_endpoint = "localhost:8080"
endpoint = f"http://{gateway_endpoint}/v2/models/{model}/infer"

# single node inference
# gateway_endpoint="localhost:32000"
# deployment_name='resnet'
# namespace = "default"
# endpoint = f"http://{gateway_endpoint}/seldon/{namespace}/{deployment_name}/v2/models/infer"


batch_test = 13

responses = []

PATH = pathlib.Path(__file__).parent.resolve()

with open(os.path.join(PATH, 'input-sample-multiple.txt'), 'r') as openfile:
    data = openfile.read()

def send_requests():
    payload = {
        "inputs": [
            {
                "name": "text_inputs",
                "shape": [1],
                "datatype": "BYTES",
                "data": [data],
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

for index, response in enumerate(responses):
    print("-"*50, f'output {index} ', "-"*50)
    inference_response = InferenceResponse.parse_raw(response.text)
    raw_json = StringRequestCodec.decode_response(inference_response)
    output = json.loads(raw_json[0])
    pp.pprint(output)
