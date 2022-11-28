from dataclasses import dataclass
from urllib import response
from barazmoon import MLServerAsyncRest
from datasets import load_dataset
import asyncio
import time

# single node mlserver
gateway_endpoint = "localhost:8080"
model = 'audio'
endpoint = f"http://{gateway_endpoint}/v2/models/{model}/infer"

# single node seldon+mlserver
# gateway_endpoint = "localhost:32000"
# deployment_name = 'audio'
# namespace = "default"
# endpoint = f"http://{gateway_endpoint}/seldon/{namespace}/{deployment_name}/v2/models/infer"

# load data
ds = load_dataset(
    "hf-internal-testing/librispeech_asr_demo",
    "clean",
    split="validation")
data = ds[0]["audio"]["array"].tolist()

http_method = 'post'
workload = 10 * [10]
data_shape = [1, len(data)]
data_type = 'audio'

start_time = time.time()

load_tester = MLServerAsyncRest(
    endpoint=endpoint,
    http_method=http_method,
    workload=workload,
    data=data,
    data_shape=data_shape,
    data_type=data_type)

responses = asyncio.run(load_tester.start())

print(f'{(time.time() - start_time):2.2}s spent in total')

import matplotlib.pyplot as plt
import numpy as np

requests = []
for sec_resps in responses:
    for resp in sec_resps:
        times = resp['timing']
        sending_time = times['sending_time']
        arrival_time = times['arrival_time']
        duration = arrival_time - sending_time
        requests.append(duration)
fig, ax = plt.subplots()
ax.plot(np.arange(len(requests)), requests)
ax.set(xlabel='request id', ylabel='arrival time - sending time (s)', title=f'MLServer rest, total time={round((time.time() - start_time))}')
ax.grid()
fig.savefig("mlserver-audio-rest.png")
plt.show()
# print(responses)
