import os
from PIL import Image
import numpy as np
from seldon_core.seldon_client import SeldonClient
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)

# single node inferline
gateway_endpoint="localhost:32000"
deployment_name = 'sum-qa'
namespace = "default"
sc = SeldonClient(
    gateway_endpoint=gateway_endpoint,
    gateway="istio",
    transport="rest",
    deployment_name=deployment_name,
    namespace=namespace)

# image = np.array(image)
response = sc.predict(
    str_data="""
After decades as a martial arts practitioner and runner, Wes "found" yoga in 2010.
He has come to appreciate that its breadth and depth provide a wonderful ballast to
steady the body and mind in today's fast-paced, technology driven lifestyle; yoga is
an antidote for stress and a pathway for deeper understanding of oneself and others.
He is an RYT 500 certified yoga instructor from the YogaWorks program, and has trained
with contemporary masters, including Ms. Maty Ezraty, co-founder of YogaWorks and a master
instructor from the Iyengar and Ashtanga traditions, as well as specialization with Mr. Bernie Clark,
a master instructor from the Yin tradition. His classes reflect these traditions,
where he combines the foundational base of precise alignment with elements of balance and focus.
These intertwine to help provide a pathway for cultivating an awareness of yourself, others, and
the world around you, as well as to create a refuge from today's fast-paced, technology-driven lifestyle.
He teaches to help others to realize the same benefit from the practice that he himself has enjoyed.
Best of all, yoga classes are just plain wonderful: they are a few moments away from life's demands
where you can simply take care of yourself physically and emotionally.
    """
)

pp.pprint(response.response['jsonData'])
