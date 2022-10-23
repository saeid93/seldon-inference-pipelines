import os
from PIL import Image
import numpy as np
from seldon_core.seldon_client import SeldonClient
from transformers import pipeline
from datasets import load_dataset
from pprint import PrettyPrinter

pp = PrettyPrinter(indent=4)

# single node inferline
gateway_endpoint="localhost:32000"
deployment_name = 'audio-qa'
namespace = "default"
sc = SeldonClient(
    gateway_endpoint=gateway_endpoint,
    gateway="istio",
    transport="rest",
    deployment_name=deployment_name,
    namespace=namespace)

ds = load_dataset(
    "hf-internal-testing/librispeech_asr_demo",
    "clean",
    split="validation")

response = sc.predict(
    data=ds[0]["audio"]["array"]
)

pp.pprint(response.response['jsonData'])
