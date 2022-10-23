from prometheus_api_client import PrometheusConnect
from prometheus_api_client.utils import parse_datetime, parse_timedelta
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import requests
PROMETHEUS = "http://localhost:30090"
prom = PrometheusConnect(url ="http://localhost:30090", disable_ssl=True)


def get_memory_usage(pod_name, name_space, container, end, need_max = False):
    PROMETHEUS = "http://localhost:30090"
    prom = PrometheusConnect(url ="http://localhost:30090", disable_ssl=True)

    query = f"container_memory_usage_bytes{{pod=~'{pod_name}.*', container='{container}', namespace='{name_space}'}}"
    if need_max:
        query = f"max_over_time(container_memory_usage_bytes{{pod=~'{pod_name}.*', container='{container}', namespace='{name_space}'}}[{end}m])"
    
    response_memory_usage = requests.get(PROMETHEUS + '/api/v1/query', params={'query' : query})
    values = response_memory_usage.json()['data']['result']
    plot_values = [[] for _ in range(len(values))]

    for val in range(len(values)):
        data = values[val]['value']
        plot_values[val].append((float(data[1])))
    return plot_values[0][0]

def get_cpu_usage(pod_name, name_space,container):
    PROMETHEUS = "http://localhost:30090"
    prom = PrometheusConnect(url ="http://localhost:30090", disable_ssl=True)

    query = f"container_cpu_usage_seconds_total{{pod=~'{pod_name}.*', namespace='{name_space}', container='{container}'}}"
    response_cpu_usage = requests.get(PROMETHEUS + '/api/v1/query', params={'query' : query})
    try:
        print(pod_name)
        values = response_cpu_usage.json()['data']['result']
    except:
        print(response_cpu_usage.json())
        exit()
    plot_values = [[] for _ in range(len(values))]

    for val in range(len(values)):
        data = values[val]['value']
        plot_values[val].append((float(data[1])))
    return plot_values[0][0]

