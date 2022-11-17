from dataclasses import dataclass
from barazmoon import MLServerAsync
import asyncio 
from datasets import load_dataset

gateway_endpoint = "localhost:32000"
deployment_name = 'mlserver-mock'
namespace = "default"
endpoint = f"http://{gateway_endpoint}/seldon/{namespace}/{deployment_name}/v2/models/infer"

ds = load_dataset(
    "hf-internal-testing/librispeech_asr_demo",
    "clean",
    split="validation")
http_method = 'post'
workload = [10] * 5 # 10 requests per second for 5 second
data = ds[0]["audio"]["array"].tolist()
data_shape = [1, len(data)]
data_type = 'audio'

load_tester = MLServerAsync(
    endpoint=endpoint,
    http_method=http_method,
    workload=workload,
    data=data,
    data_shape=data_shape,
    data_type=data_type)

responses = asyncio.run(load_tester.start())

# print(responses)
