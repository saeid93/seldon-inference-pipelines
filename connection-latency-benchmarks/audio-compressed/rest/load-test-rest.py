from dataclasses import dataclass
from urllib import response
from barazmoon import MLServerAsyncRest
from datasets import load_dataset
import asyncio
import time
import json


# endpoint = 'http://127.0.0.1:8000'

# load data
ds = load_dataset(
    "hf-internal-testing/librispeech_asr_demo",
    "clean",
    split="validation")
data = ds[0]["audio"]["array"].tolist()
# data = ds[0]["audio"]["array"]

http_method = 'post'
load = 2
test_duration = 30
variant = 0
platform = 'seldon'
workload = [load] * test_duration
data_shape = [1, len(data)]
data_type = 'audio-base64'
mode = 'equal' # options - step, equal, exponential

# single node inference
if platform == 'seldon':
    gateway_endpoint = "localhost:32000"
    deployment_name = 'latency-connection-rest-audio-base64'
    model = 'mock-one'
    namespace = "default"
    endpoint = f"http://{gateway_endpoint}/seldon/{namespace}/{deployment_name}/v2/models/mock-one/infer"
elif platform == 'mlserver':
    gateway_endpoint = "localhost:8080"
    model = 'mock-one'
    endpoint = f"http://{gateway_endpoint}/v2/models/{model}/infer"
elif platform == 'fastapi':
    endpoint = "http://127.0.0.1:8000"

start_time = time.time()

load_tester = MLServerAsyncRest(
    endpoint=endpoint,
    http_method=http_method,
    workload=workload,
    data=data,
    mode=mode,
    data_shape=data_shape,
    data_type=data_type)

responses = asyncio.run(load_tester.start())

print(f'{(time.time() - start_time):2.2}s spent in total')

import matplotlib.pyplot as plt
import numpy as np

# Through away initial seconds results
responses = responses[3:]

# # roundtrip latency
# roundtrip_lat = []
# for sec_resps in responses:
#     for resp in sec_resps:
#         request_times = resp['times']['request']
#         sending_time = request_times['arrival'] - request_times['sending']
#         roundtrip_lat.append(sending_time)
# fig, ax = plt.subplots()
# ax.plot(np.arange(len(roundtrip_lat)), roundtrip_lat)
# ax.set(xlabel='request id', ylabel='roundtrip latency (s)', title=f'roundtrip latency, total time={round((time.time() - start_time))}')
# ax.grid()
# fig.savefig(f"rest-compressed-audio-{platform}_variant_{variant}-roundtrip_lat-load-{load}-test_duration-{test_duration}.png")
# plt.show()

# # sending time
# start_times = []
# for sec_resps in responses:
#     for resp in sec_resps:
#         times = resp['times']['request']
#         sending_time = times['sending'] - start_time
#         start_times.append(sending_time)
# fig, ax = plt.subplots()
# ax.plot(np.arange(len(start_times)), start_times)
# ax.set(xlabel='request id', ylabel='sending time (s)', title=f'load tester sending time, total time={round((time.time() - start_time))}')
# ax.grid()
# fig.savefig(f"rest-compressed-audio-{platform}_variant_{variant}-sending_time-load-{load}-test_duration-{test_duration}.png")
# plt.show()

# # server arrival time
# server_arrival_time = []
# for sec_resps in responses:
#     for resp in sec_resps:
#         times = resp['times']
#         server_recieving_time = times['models'][model]['arrival'] - start_time
#         server_arrival_time.append(server_recieving_time)
# fig, ax = plt.subplots()
# ax.plot(np.arange(len(server_arrival_time)), server_arrival_time)
# ax.set(xlabel='request id', ylabel='server arrival time (s)', title=f'Server recieving time of requests, total time={round((time.time() - start_time))}')
# ax.grid()
# fig.savefig(f"rest-compressed-audio-{platform}_variant_{variant}-server_arrival_time_from_start-load-{load}-test_duration-{test_duration}.png")
# plt.show()

# server arrival latency
server_arrival_latency = []
for sec_resps in responses:
    for resp in sec_resps:
        times = resp['times']
        server_recieving_time = times['models'][model]['arrival'] - times['request']['sending']
        server_arrival_latency.append(server_recieving_time)
fig, ax = plt.subplots()
ax.plot(np.arange(len(server_arrival_latency)), server_arrival_latency)
ax.set(xlabel='request id', ylabel='server arrival latency (s)', title=f'Server recieving latency, total time={round((time.time() - start_time))}')
ax.grid()
fig.savefig(f"rest-compressed-audio-{platform}_variant_{variant}-server_recieving_latency-load-{load}-test_duration-{test_duration}.png")
plt.show()

print(f"{np.average(server_arrival_latency)}=")
