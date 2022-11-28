from dataclasses import dataclass
from barazmoon import MLServerAsyncRest
import asyncio 
from datasets import load_dataset

gateway_endpoint = "localhost:32000"
deployment_name = 'mock-pipeline'
namespace = "default"
endpoint = f"http://{gateway_endpoint}/seldon/{namespace}/{deployment_name}/v2/models/infer"

ds = load_dataset(
    "hf-internal-testing/librispeech_asr_demo",
    "clean",
    split="validation")
http_method = 'post'
workload = [10, 7, 4, 12, 15]
data = ds[0]["audio"]["array"].tolist()
data_shape = [1, len(data)]
data_type = 'audio'

load_tester = MLServerAsyncRest(
    endpoint=endpoint,
    http_method=http_method,
    workload=workload,
    data=data,
    data_shape=data_shape,
    data_type=data_type)

responses = asyncio.run(load_tester.start())

print(responses)
