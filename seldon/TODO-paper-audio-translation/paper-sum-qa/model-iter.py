import os
from plistlib import load
from re import TEMPLATE
from typing import Any, Dict
from PIL import Image
import numpy as np
from seldon_core.seldon_client import SeldonClient
from jinja2 import Environment, FileSystemLoader
import time
import subprocess
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)

PATH = "/home/cc/infernece-pipeline-joint-optimization/pipelines/23-pipelines-prototype/nlp/seldon-core-version"
CHECK_TIMEOUT = 2
RETRY_TIMEOUT = 60
DELETE_WAIT = 10
TEMPLATE = "nlp"

inputs = """
Après des décennies en tant que pratiquant d'arts martiaux et coureur, Wes a "trouvé" le yoga en 2010.
Il en est venu à apprécier que son ampleur et sa profondeur fournissent un merveilleux lest pour stabiliser
le corps et l'esprit dans le style de vie rapide et axé sur la technologie d'aujourd'hui ;
le yoga est un antidote au stress et une voie vers une meilleure compréhension de soi et des autres.
Il est instructeur de yoga certifié RYT 500 du programme YogaWorks et s'est formé avec des maîtres contemporains,
dont Mme Maty Ezraty, co-fondatrice de YogaWorks et maître instructeur des traditions Iyengar et Ashtanga,
ainsi qu'une spécialisation avec M. Bernie. Clark, un maître instructeur de la tradition Yin.
Ses cours reflètent ces traditions, où il combine la base fondamentale d'un alignement précis avec des éléments
d'équilibre et de concentration. Ceux-ci s'entremêlent pour aider à fournir une voie pour cultiver une conscience
de vous-même, des autres et du monde qui vous entoure, ainsi que pour créer un refuge contre le style de vie rapide
et axé sur la technologie d'aujourd'hui. Il enseigne à aider les autres à réaliser le même bénéfice de la pratique dont il a lui-même bénéficié.
Mieux encore, les cours de yoga sont tout simplement merveilleux :
ils sont à quelques instants des exigences de la vie où vous pouvez simplement prendre soin de vous physiquement et émotionnellement.
    """

def load_test(
    pipeline_name: str,
    inputs: Dict[str, Any],
    n_items: int
    ):
    # TODO change here
    # single node inferline
    gateway_endpoint="localhost:32000"
    deployment_name = pipeline_name 
    namespace = "default"

    sc = SeldonClient(
        gateway_endpoint=gateway_endpoint,
        gateway="istio",
        transport="rest",
        deployment_name=deployment_name,
        namespace=namespace)

    time.sleep(CHECK_TIMEOUT)
    response = sc.predict(
        str_data=inputs
    )

    if response.success:
        pp.pprint(response.response['jsonData'])
    else:
        pp.pprint(response.msg)

def setup_pipeline(
    node_1_model: str,
    node_2_model: str,
    node_3_model: str,   
    template: str,
    pipeline_name: str):
    svc_vars = {
        "node_1_variant": node_1_model,
        "node_2_variant": node_2_model,
        "node_3_variant": node_3_model,        
        "pipeline_name": pipeline_name}
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

# check all the possible combination
node_1_models = [
    'dinalzein/xlm-roberta-base-finetuned-language-identification']

node_2_models = [
    'Helsinki-NLP/opus-mt-fr-en']

node_3_models = [
    'sshleifer/distilbart-cnn-12-6']

for node_1_model in node_1_models:
    for node_2_model in node_2_models:
        for node_3_model in node_3_models:
            pipeline_name = node_1_model[:8].lower() + "-" +\
                node_2_model[:8].lower() + "-" + node_3_model[:8].lower()
            start_time = time.time()
            while True:
                setup_pipeline(
                    node_1_model=node_1_model,
                    node_2_model=node_2_model,
                    node_3_model=node_3_model,                    
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
            load_test(pipeline_name=pipeline_name, inputs=inputs, n_items=1)

            time.sleep(DELETE_WAIT)

            print("operation done, deleting the pipeline ...")
            remove_pipeline(pipeline_name=pipeline_name)
            print('pipeline successfuly deleted')
