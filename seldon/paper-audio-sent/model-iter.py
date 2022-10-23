"""
Iterate through all possible combination
of models and servers
"""

import os
from plistlib import load
import yaml
from re import TEMPLATE
from typing import Any, Dict
from seldon_core.seldon_client import SeldonClient
from jinja2 import Environment, FileSystemLoader
import time
from prom import get_cpu_usage, get_memory_usage
import subprocess
from datasets import load_dataset
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)


PATH = "/home/cc/infernece-pipeline-joint-optimization/pipelines/seldon-prototype/paper-audio-sent/seldon-core-version"
PIPELINES_MODELS_PATH = "/home/cc/infernece-pipeline-joint-optimization/data/pipeline-test-meta" # TODO fix be moved to utilspr
DATABASE = "/home/cc/infernece-pipeline-joint-optimization/data/pipeline"
CHECK_TIMEOUT = 60
RETRY_TIMEOUT = 90
DELETE_WAIT = 45
LOAD_TEST_WAIT = 60
TRIAL_END_WAIT = 60
TEMPLATE = "audio"
CONFIG_FILE = "paper-audio-sent"
save_path = os.path.join(DATABASE, "audio-data-soso")
if not os.path.exists(save_path):
    os.makedirs(save_path)


ds = load_dataset(
    "hf-internal-testing/librispeech_asr_demo",
    "clean",
    split="validation")

inputs=ds[0]["audio"]["array"]


def change_names(names):
    return_names = []
    for name in names:
        return_names.append(name.replace("_", "-"))
    return return_names


def extract_node_timer(json_data : dict):
    keys = list(json_data.keys())
    nodes = []
    sir_names = ["arrival_"]
    for name in sir_names:
        for key in keys:
            if name in key:
                nodes.append(key.replace(name, ""))

    return_nodes = change_names(nodes)
    return_timer = {}
    for node in nodes:
        return_timer[node] = json_data["serving_" + node] - json_data["arrival_" + node]
    e2e_lats = json_data[keys[-1]] - json_data[keys[0]]

    return return_nodes, return_timer, e2e_lats
    
def load_test(
    pipeline_name: str,
    inputs: Dict[str, Any],
    node_1_model, 
    node_2_model,
    n_items: int,
    n_iters = 15
    ):
    start = time.time()
    gateway_endpoint="localhost:32000"
    deployment_name = pipeline_name 
    namespace = "default"
    num_nodes = pipeline_name.split("-").__len__()
    e2e_lats = []
    node_latencies = [[] for _ in range(num_nodes)]
    cpu_usages = [[] for _ in range(num_nodes) ]
    memory_usages = [[] for _ in range(num_nodes) ]
    sc = SeldonClient(
        gateway_endpoint=gateway_endpoint,
        gateway="istio",
        transport="rest",
        deployment_name=deployment_name,
        namespace=namespace)

    time.sleep(CHECK_TIMEOUT)
    for iter in range(n_iters):
        response = sc.predict(
            data=inputs
        )

        if response.success:
            json_data_timer = response.response['jsonData']['time']
            return_nodes, return_timer, e2e_lat = extract_node_timer(json_data_timer)
            for i , name in enumerate(return_nodes):
                cpu_usages[i].append(get_cpu_usage(pipeline_name, "default", name))
                memory_usages[i].append(get_memory_usage(pipeline_name, "default", name, 1))
                e2e_lats.append(e2e_lat)
            for i, time_ in enumerate(return_timer.keys()):
                node_latencies[i].append(return_timer[time_])

        else:
            pp.pprint(response.msg)
        print(iter)
    time.sleep(CHECK_TIMEOUT)
    total_time = int((time.time() - start)//60)
    for i , name in enumerate(return_nodes):
        cpu_usages[i].append(get_cpu_usage(pipeline_name, "default", name))
        memory_usages[i].append(get_memory_usage(pipeline_name, "default", name, total_time, True))
    models = node_1_model + "*" + node_2_model + "*"
    with open(save_path+"/cpu.txt", "a") as cpu_file:
        cpu_file.write(f"usage of {models} {pipeline_name} is {cpu_usages} \n")

    with open(save_path+"/memory.txt", 'a') as memory_file:
        memory_file.write(f"usage of {models} {pipeline_name} is {memory_usages} \n")


    with open(save_path+"/node-latency.txt", "a") as infer:
        infer.write(f"lats of {models} {pipeline_name} is {node_latencies} \n")
    
    with open(save_path+"/ee.txt", "a") as s:
        s.write(f"eelat of {models} {pipeline_name} is {e2e_lats} \n")
    
def setup_pipeline(
    node_1_model: str,
    node_2_model: str, 
    template: str,
    pipeline_name: str):
    svc_vars = {
        "node_1_variant": node_1_model,
        "node_2_variant": node_2_model,        
        "pipeline_name": pipeline_name,
        "cpu_limits": 6,
        "cpu_requests": 6
        }
    environment = Environment(
        loader=FileSystemLoader(os.path.join(
            PATH, "templates/")))
    svc_template = environment.get_template(f"{template}.yaml")
    content = svc_template.render(svc_vars)
    command = f"""cat <<EOF | kubectl apply -f -
{content}
    """
    os.system(command)

def remove_pipeline(pipeline_name):
    os.system(f"kubectl delete seldondeployment {pipeline_name} -n default")

config_file_path = os.path.join(
    PIPELINES_MODELS_PATH, f"{CONFIG_FILE}.yaml")
with open(config_file_path, 'r') as cf:
    config = yaml.safe_load(cf)

node_1_models = config['node_1']
node_2_models = config['node_2']

def prune_name(name, len):
    forbidden_strs = ['facebook', '/', 'huggingface', '-']
    for forbidden_str in forbidden_strs:
        name = name.replace(forbidden_str, '')
    name = name.lower()
    name = name[:len]
    return name

for node_1_model in node_1_models:
    for node_2_model in node_2_models:
        pipeline_name = prune_name(node_1_model, 8) + "-" +\
            prune_name(node_2_model, 8)
        start_time = time.time()
        while True:
            setup_pipeline(
                node_1_model=node_1_model,
                node_2_model=node_2_model,
                template=TEMPLATE, pipeline_name=pipeline_name)
            time.sleep(CHECK_TIMEOUT)
            command = ("kubectl rollout status deploy/$(kubectl get deploy"
                    f" -l seldon-deployment-id={pipeline_name} -o"
                    " jsonpath='{.items[0].metadata.name}')")
            time.sleep(CHECK_TIMEOUT)
            p = subprocess.Popen(command, shell=True)
            try:
                p.wait(RETRY_TIMEOUT)
                break
            except subprocess.TimeoutExpired:
                p.kill()
                print("corrupted pipeline, should be deleted ...")
                remove_pipeline(pipeline_name=pipeline_name)
                print('waiting to delete ...')
                time.sleep(DELETE_WAIT)

        print('starting the load test ...\n')
        load_test(pipeline_name=pipeline_name, inputs=inputs, node_1_model=node_1_model, node_2_model=node_2_model, n_items=1)

        time.sleep(DELETE_WAIT)

        print("operation done, deleting the pipeline ...")
        remove_pipeline(pipeline_name=pipeline_name)
        print('pipeline successfuly deleted')
