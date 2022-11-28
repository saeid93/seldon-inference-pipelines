import requests
import json
from mlserver.types import InferenceResponse
from mlserver.codecs.string import StringRequestCodec
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)

# single node mlserver
# gateway_endpoint="localhost:32000"
# deployment_name = 'nlp-qa'
# namespace = "default"
# endpoint = f"http://{gateway_endpoint}/seldon/{namespace}/{deployment_name}/v2/models/infer"

# single node seldon+mlserver
gateway_endpoint="localhost:8080"
model='nlp-qa'
endpoint = f"http://{gateway_endpoint}/v2/models/{model}/infer"

def send_requests(endpoint, data):
    payload = {
        "inputs": [
            {
            "name": "text_inputs",
            "shape": [1],
            "datatype": "BYTES",
            "data": data,
            "parameters": {
                "content_type": "str"
            }
            }
        ]
    }
    response = requests.post(endpoint, json=payload)
    return response

data=['{"time": {"arrival_audio": 1664649974.6980114,'
      ' "serving_audio": 1664649974.9401753}, "output":'
      ' {"text": "mister quilter is the apostle of the middle'
      ' classes and we are glad to welcome his gospel"}}']


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
