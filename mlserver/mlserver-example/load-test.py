from dataclasses import dataclass
# from barazmoon import MLServerBarAzmoon


import numpy as np
import matplotlib.pyplot as plt

def preprocess(contents):
    data = contents[0].split(', ')
    for d in range(len(data)):
        if '[' in data[d]:
            data[d] = data[d].replace('[','')
        if ']' in data[d]:
            data[d] = data[d].replace(']','')
        data[d] = int(data[d])
    return data

def timer(data):
    starter = 0
    end = 0
    flag_start = False
    for d in range(len(data)):
        if data[d] > 0 and flag_start == False:
            starter = d
            flag_start = True
        
        if sum(data[d:]) == 0:
            end = d
            break
    return starter, end

with open('../../../twitter-trace-preprocess/workload.txt') as f:
    contents_workload = f.readlines()

data_workload = preprocess(contents_workload)
sw, ew = timer(data_workload)

workload = data_workload[sw:ew]

print(workload)




# gateway_endpoint = "localhost:32000"
# deployment_name = 'custom-mlserver'
# namespace = "default"
# endpoint = f"http://{gateway_endpoint}/seldon/{namespace}/{deployment_name}/v2/models/infer"

# http_method = 'post'
# workload = [10, 7, 4, 12, 15]
# data = [1, 2]
# data_shape = [2, 1]
# data_type = 'example'


# load_tester = MLServerBarAzmoon(
#     endpoint=endpoint,
#     http_method=http_method,
#     workload=workload,
#     data=data,
#     data_shape=data_shape,
#     data_type='example')
# load_tester.start()