import os
import pathlib
from PIL import Image
import numpy as np
from mlserver.types import InferenceResponse
from mlserver.codecs.string import StringRequestCodec
import requests
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)
import threading
import json

# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True

def image_loader(folder_path, image_name):
    image = Image.open(
        os.path.join(folder_path, image_name))
    # if there was a need to filter out only color images
    # if image.mode == 'RGB':
    #     pass
    return image

# os.system('sudo umount -l ~/my_mounting_point')
# os.system('cc-cloudfuse mount ~/my_mounting_point')

# data_folder_path = '/home/cc/my_mounting_point/datasets'
# dataset_folder_path = os.path.join(
#     data_folder_path, 'ILSVRC/Data/DET/test'
# )
# classes_file_path = os.path.join(
#     data_folder_path, 'imagenet_classes.txt'
# )
# with open(classes_file_path) as f:
#     classes = [line.strip() for line in f.readlines()]

# image_names = os.listdir(dataset_folder_path)
# image_names.sort()

# num_loaded_images = 20

# images = {
#     image_name: image_loader(
#         dataset_folder_path, image_name) for image_name in image_names[
#             :num_loaded_images]}

# input_data = images["ILSVRC2017_test_00000009.JPEG"]
# input_data_shape = [1] + list(np.shape(input_data))

PATH = pathlib.Path(__file__).parent.resolve()

input_data = image_loader(PATH, 'input-sample.JPEG')

with open(os.path.join(
    PATH, 'input-sample-shape.json'), 'r') as openfile:
    input_data_shape = json.load(openfile)
    input_data_shape = input_data_shape['data_shape']

input_data_shape = [1] + list(np.shape(input_data))

gateway_endpoint="localhost:32000"
deployment_name = 'yolo'
namespace = "default"
endpoint = f"http://{gateway_endpoint}/seldon/{namespace}/{deployment_name}/v2/models/infer"

# gateway_endpoint="localhost:8080"
# endpoint = f"http://{gateway_endpoint}/v2/models/yolo/infer"

batch_test = 6
responses = []

def send_requests():
    input_ins = {
        "name": "parameters-np",
        "datatype": "INT32",
        "shape": input_data_shape,
        "data": np.array(input_data).tolist(),
        "parameters": {
            "content_type": "np"
            }
        }
    payload = {
    "inputs": [input_ins]
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

for index, output in enumerate(outputs):
    print("-"*50, f' {index} ', "-"*50)
    output = list(map(lambda l: np.array(l), output['output']['person']))
    pp.pprint(len(output))
