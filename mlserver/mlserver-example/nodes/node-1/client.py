import requests
from pprint import PrettyPrinter
import threading

pp = PrettyPrinter(indent=4)

input_ins = {
    "name": "parameters-np",
    "datatype": "FP32",
    "shape": [2, 1],
    "data": [12, 43],
    "parameters": {
        "content_type": "np"
        }
    }

endpoint = "http://localhost:32000/seldon/default/custom-mlserver-node-one/v2/models/infer"
# endpoint = "http://localhost:8080/v2/models/node-1/infer"
# response = requests.post(endpoint, json=payload)

batch_test = 20
payload = {
    "inputs": [input_ins]
}

responses = []
def send_requests():
    response = requests.post(endpoint, json=payload)
    # print('\n')
    # print('-' * 50)
    # pp.pprint(response.json())
    responses.append(response)
    return response

# for i in range(batch_test):
#     send_requests()

thread_pool = []

for i in range(batch_test):
    t = threading.Thread(target=send_requests)
    t.start()
    thread_pool.append(t)

for t in thread_pool:
    t.join()


pp.pprint(list(map(lambda l:l.json(), responses)))