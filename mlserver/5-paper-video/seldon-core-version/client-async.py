import os
from PIL import Image
import numpy as np
from mlserver.types import InferenceResponse
from mlserver.codecs.string import StringRequestCodec
import requests
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)
import threading
import json

# PIPELINES_MODELS_PATH = "/home/cc/infernece-pipeline-joint-optimization/data/sample-image/"
# dataset_folder_path = PIPELINES_MODELS_PATH

os.system('sudo umount -l ~/my_mounting_point')
os.system('cc-cloudfuse mount ~/my_mounting_point')

data_folder_path = '/home/cc/my_mounting_point/datasets'
dataset_folder_path = os.path.join(
    data_folder_path, 'ILSVRC/Data/DET/test'
)
classes_file_path = os.path.join(
    data_folder_path, 'imagenet_classes.txt'
)
with open(classes_file_path) as f:
    classes = [line.strip() for line in f.readlines()]

image_names = os.listdir(dataset_folder_path)
image_names.sort()

num_loaded_images = 20

def image_loader(folder_path, image_name):
    image = Image.open(
        os.path.join(folder_path, image_name))
    # if there was a need to filter out only color images
    # if image.mode == 'RGB':
    #     pass
    return image

images = {
    image_name: image_loader(
        dataset_folder_path, image_name) for image_name in image_names[
            :num_loaded_images]}

input_data = images["ILSVRC2017_test_00000009.JPEG"]
input_data_shape = [1] + list(np.shape(input_data))

gateway_endpoint="localhost:32000"
deployment_name = 'video'
namespace = "default"

endpoint = f"http://{gateway_endpoint}/seldon/{namespace}/{deployment_name}/v2/models/infer"


batch_test = 1
responses = []

def send_requests():
    payload = {
        "inputs":[
            {
                "name": "parameters-np",
                "datatype": "INT32",
                "shape": input_data_shape,
                "data": np.array(input_data).tolist(),
                "parameters": {
                    "content_type": "np"
                    }
            }]
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