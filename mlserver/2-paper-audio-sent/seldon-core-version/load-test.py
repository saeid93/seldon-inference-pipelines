from dataclasses import dataclass
from barazmoon import MLServerBarAzmoon
from datasets import load_dataset

gateway_endpoint = "localhost:32000"
deployment_name = 'audio-sent'
namespace = "default"
endpoint = f"http://{gateway_endpoint}/seldon/{namespace}/{deployment_name}/v2/models/infer"

ds = load_dataset(
    "hf-internal-testing/librispeech_asr_demo",
    "clean",
    split="validation")
http_method = 'post'
workload = [10, 7, 4, 12, 15] # 10 reqs at second 1, 7 reqs at second 2, ...
data = ds[0]["audio"]["array"].tolist()
data_shape = [1, len(data)]
data_type = 'audio'

load_tester = MLServerBarAzmoon(
    endpoint=endpoint,
    http_method=http_method,
    workload=workload,
    data=data,
    data_shape=data_shape,
    data_type=data_type)

load_tester.start()