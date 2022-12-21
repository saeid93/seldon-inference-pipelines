import os
import pathlib
from PIL import Image
import numpy as np
from mlserver.types import InferenceResponse
from mlserver.codecs.string import StringRequestCodec
import requests
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)
import json
import base64

def image_loader(folder_path, image_name):
    image = Image.open(
        os.path.join(folder_path, image_name))
    # if there was a need to filter out only color images
    # if image.mode == 'RGB':
    #     pass
    return image

model_name = 'mock-one'
gateway_endpoint="localhost:8080"
endpoint = f"http://{gateway_endpoint}/v2/models/{model_name}/infer"

PATH = pathlib.Path(__file__).parent.resolve()
input_data = image_loader(PATH, 'input-sample.JPEG')
with open(os.path.join(
    PATH, 'input-sample-shape.json'), 'r') as openfile:
    input_data_shape = json.load(openfile)
    input_data_shape = input_data_shape['data_shape']
images = {}
images['inpue-sample'] = input_data
input_data = images
inputs = [input_data] * 1

def encode_to_bin(im_arr):
    im_bytes = im_arr.tobytes()
    im_base64 = base64.b64encode(im_bytes)
    input_dict = im_base64.decode()
    return input_dict

def send_requests(endpoint, image):
    input_ins = {
        "name": "parameters-np",
        "datatype": "BYTES",
        "shape": list(np.shape(image)),
        "data": encode_to_bin(np.array(image)),
        "parameters": {
            "content_type": "np",
            "dtype": "u1"
            }
        }
    payload = {
    "inputs": [input_ins]
    }
    response = requests.post(endpoint, json=payload)
    return response

# sync version
results = {}
for image in inputs:
    for image_name, image in image.items():
        response = send_requests(endpoint, image)
        inference_response = InferenceResponse.parse_raw(response.text)
        raw_json = StringRequestCodec.decode_response(inference_response)
        output = json.loads(raw_json[0])
        results[image_name] = output

for image_name, output in results.items():
    print("-"*50, f' {image_name} ', "-"*50)
    pp.pprint(output)
