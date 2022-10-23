import os
import pathlib
from barazmoon import MLServerBarAzmoon


gateway_endpoint = "localhost:32000"
deployment_name = 'resnet-human'
namespace = "default"
endpoint = f"http://{gateway_endpoint}/seldon/{namespace}/{deployment_name}/v2/models/infer"

PATH = pathlib.Path(__file__).parent.resolve()

with open(os.path.join(PATH, 'input-sample.txt'), 'r') as openfile:
    data = openfile.read()

http_method = 'post'
workload = [6]
data_shape = [1]
data_type = 'text'

load_tester = MLServerBarAzmoon(
    endpoint=endpoint,
    http_method=http_method,
    workload=workload,
    data=[data],
    data_shape=data_shape,
    data_type=data_type)

load_tester.start()

print(load_tester.get_responses())
