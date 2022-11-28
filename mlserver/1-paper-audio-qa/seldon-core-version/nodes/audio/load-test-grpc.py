from barazmoon import MLServerAsyncGrpc
from datasets import load_dataset
import asyncio
import time

# single node mlserver
# endpoint = "localhost:8081"
# model = 'audio'
# metadata = []

# single node seldon+mlserver
endpoint = "localhost:32000"
deployment_name = 'audio'
model = 'audio'
namespace = "default"
metadata = [("seldon", deployment_name), ("namespace", namespace)]

# load data
ds = load_dataset(
    "hf-internal-testing/librispeech_asr_demo",
    "clean",
    split="validation")
data = ds[0]["audio"]["array"].tolist()

workload = [10] * 10
data_shape = [1, len(data)]
data_type = 'audio'

start_time = time.time()

load_tester = MLServerAsyncGrpc(
    endpoint=endpoint,
    metadata=metadata,
    workload=workload,
    model=model,
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
ax.set(xlabel='request id', ylabel='arrival time - sending time (s)', title=f'MLServer grpc, total time={round((time.time() - start_time))}')
ax.grid()
fig.savefig("mlserver-audio-grpc.png")
plt.show()
# print(responses)
