import os
from PIL import Image
import numpy as np
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)
from barazmoon import MLServerBarAzmoon

gateway_endpoint="localhost:32000"
deployment_name = 'sum-qa'
namespace = "default"
endpoint = f"http://{gateway_endpoint}/seldon/{namespace}/{deployment_name}/v2/models/infer"


workload = [10, 7, 4, 12, 15]
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

data = images["ILSVRC2017_test_00000009.JPEG"]
data_shape = [1] + list(np.shape(data))
data = np.array(data).tolist()
http_method = 'post'
data_type = 'image'

load_tester = MLServerBarAzmoon(
    endpoint=endpoint,
    http_method=http_method,
    workload=workload,
    data=data,
    data_shape=data_shape,
    data_type=data_type)

load_tester.start()