import os
import PIL
from PIL import Image
from typing import Dict

import numpy as np

from seldon_core.seldon_client import SeldonClient

data_folder_path = '/home/cc/my_mounting_point/datasets'
dataset_folder_path = os.path.join(
    data_folder_path, 'ILSVRC/Data/DET/test'
)
classes_file_path = os.path.join(
    data_folder_path, 'imagenet_classes.txt'
)

image_names = os.listdir(dataset_folder_path)
image_names.sort()
with open(classes_file_path) as f:
    classes = [line.strip() for line in f.readlines()]
num_loaded_images = 5

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

# single node inferline
gateway_endpoint="localhost:32000"
deployment_name = 'inferline-preprocess'
namespace = "default"
sc = SeldonClient(
    gateway_endpoint=gateway_endpoint,
    gateway="istio",
    transport="rest",
    deployment_name=deployment_name,
    namespace=namespace)

results = {}
for image_name, image in images.items():
    image = np.array(image)
    response = sc.predict(
        data=image
    )
    results[image_name] = response

for image_name, response in results.items():
    print(f"\nimage name: {image_name}")
    if response.success:

        request_path = response.response['meta']['requestPath'].keys()
        pipeline_response = response.response['jsonData']
        print(f"request path: {request_path}")
        print(f"response: \n{pipeline_response}")
    else:
        print(f"{image_name} -> {response.msg}")
