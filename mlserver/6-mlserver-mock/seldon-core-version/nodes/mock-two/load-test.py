import os
import pathlib
from barazmoon import MLServerAsyncRest
import asyncio
# from barazmoon import queue
# from barazmoon import responses


gateway_endpoint = "localhost:32000"
deployment_name = 'mock-two'
namespace = "default"
endpoint = f"http://{gateway_endpoint}/seldon/{namespace}/{deployment_name}/v2/models/infer"

PATH = pathlib.Path(__file__).parent.resolve()

with open(os.path.join(PATH, 'input-sample.txt'), 'r') as openfile:
    data = openfile.read()

http_method = 'post'
workload = [3, 4, 2]
data_shape = [1]
data_type = 'text'

load_tester = MLServerAsyncRest(
    endpoint=endpoint,
    http_method=http_method,
    workload=workload,
    data=data,
    data_shape=data_shape,
    data_type=data_type)

responses = asyncio.run(load_tester.start())

print(responses)
