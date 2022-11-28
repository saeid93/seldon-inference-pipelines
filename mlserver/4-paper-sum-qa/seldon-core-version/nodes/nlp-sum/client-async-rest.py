import os
import pathlib
import json
import requests
import threading
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)
from mlserver.types import InferenceResponse
from mlserver.codecs.string import StringRequestCodec
from pprint import PrettyPrinter

pp = PrettyPrinter(indent=1)

# single node inference
# model = 'nlp-sum'
# gateway_endpoint = "localhost:8080"
# endpoint = f"http://{gateway_endpoint}/v2/models/{model}/infer"

# single node inferline
gateway_endpoint="localhost:32000"
deployment_name = 'nlp-sum'
namespace = "default"
endpoint = f"http://{gateway_endpoint}/seldon/{namespace}/{deployment_name}/v2/models/infer"


batch_test = 6

responses = []

PATH = pathlib.Path(__file__).parent.resolve()

with open(os.path.join(PATH, 'input-sample.txt'), 'r') as openfile:
    data = openfile.read()

data = [data]


def send_requests():
    payload = {
        "inputs": [
            {
                "name": "text_inputs",
                "shape": [1],
                "datatype": "BYTES",
                "data": data,
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