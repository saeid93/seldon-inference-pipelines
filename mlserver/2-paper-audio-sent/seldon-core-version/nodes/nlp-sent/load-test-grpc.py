import os
import pathlib
import asyncio
from barazmoon import MLServerAsyncGrpc
import asyncio

# single node mlserver
endpoint = "localhost:8081"
model = 'nlp-sent'
metadata = []

# single node seldon+mlserver
# endpoint = "localhost:32000"
# deployment_name = 'nlp-sent'
# model = 'nlp-sent'
# namespace = "default"
# metadata = [("seldon", deployment_name), ("namespace", namespace)]

PATH = pathlib.Path(__file__).parent.resolve()

with open(os.path.join(PATH, 'input-sample.txt'), 'r') as openfile:
    data = openfile.read()

workload = [2, 2, 2]
data_shape = [1]
data_type = 'text'

load_tester = MLServerAsyncGrpc(
    endpoint=endpoint,
    metadata=metadata,
    workload=workload,
    model=model,
    data=data,
    data_shape=data_shape,
    data_type=data_type)

responses = asyncio.run(load_tester.start())

print(responses)
