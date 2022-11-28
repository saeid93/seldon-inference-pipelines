import os
import pathlib
from PIL import Image
import numpy as np
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)
from barazmoon import MLServerAsyncRest
import asyncio
import json

# gateway_endpoint="localhost:32000"
# deployment_name = 'yolo'
# namespace = "default"
# endpoint = f"http://{gateway_endpoint}/seldon/{namespace}/{deployment_name}/v2/models/infer"

gateway_endpoint="localhost:8080"
endpoint = f"http://{gateway_endpoint}/v2/models/yolo/infer"

workload = [10, 7, 4, 12, 15]

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

# data = images["ILSVRC2017_test_00000009.JPEG"]
# data_shape = [1] + list(np.shape(data))
# data = np.array(data).tolist()
# input_data_shape = [1] + list(np.shape(input_data))


PATH = pathlib.Path(__file__).parent.resolve()
input_data = image_loader(PATH, 'input-sample.JPEG')
with open(os.path.join(
    PATH, 'input-sample-shape.json'), 'r') as openfile:
    input_data_shape = json.load(openfile)
    input_data_shape = input_data_shape['data_shape']
input_data = np.array(input_data).flatten().tolist()


http_method = 'post'
data_type = 'image'


load_tester = MLServerAsyncRest(
    endpoint=endpoint,
    http_method=http_method,
    workload=workload,
    data=input_data,
    data_shape=input_data_shape,
    data_type=data_type)

responses = asyncio.run(load_tester.start())

# print(responses)
