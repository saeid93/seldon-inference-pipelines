import requests
from pprint import PrettyPrinter
import threading

pp = PrettyPrinter(indent=4)

input_ins = {
    "name": "parameters-np",
    "datatype": "FP32",
    "shape": [2, 1],
    "data": [1, 2],
    "parameters": {
        "content_type": "np"
        }
    }

endpoint = "http://localhost:32000/seldon/default/custom-mlserver/v2/models/infer"

# batch_test = 1
# payload = {
#     "inputs": [input_ins]
# }

# def send_requests():
#     response = requests.post(endpoint, json=payload)
#     print(response)

# send_requests()

responses = []

batch_test = 50
payload = {
    "inputs": [input_ins]
}
import time
responses = []
def send_requests():
    response = requests.post(endpoint, json=payload)
    responses.append(response)
    print(response)
    print(time.time())
    return response


thread_pool = []

for i in range(batch_test):
    t = threading.Thread(target=send_requests)
    t.start()
    thread_pool.append(t)

for t in thread_pool:
    t.join()


pp.pprint(list(map(lambda l:l.json(), responses)))
