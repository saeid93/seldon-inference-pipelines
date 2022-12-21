import time
from typing import List, Tuple
import numpy as np
from multiprocessing import Process, active_children #, Lock
import asyncio
from aiohttp import ClientSession, ClientTimeout
from aiohttp import TCPConnector
from numpy.random import default_rng
import aiohttp
import asyncio
import grpc
from mlserver.codecs.string import StringRequestCodec
import mlserver.grpc.dataplane_pb2_grpc as dataplane
import mlserver.grpc.converters as converters
import json
# from multiprocessing import Queue


# from multiprocessing import Queue

# MAX_QUEUE_SIZE = 1000000
# queue = Queue(MAX_QUEUE_SIZE)
# lock = Lock()

# TCP_CONNECTIONS = 500
# connector = TCPConnector(limit=TCP_CONNECTIONS)

TIMEOUT = 20 * 60

# ============= Async + Process based load tester =============

# TODO problems:
# 1. high process load -> not closing finished processes on time
# 2. having a way to save the responses recieved from the load-tester

class BarAzmoonProcess:
    def __init__(
        self, *, workload: List[int], endpoint: str,
        http_method = "get", **kwargs):
        self.endpoint = endpoint
        self.http_method = http_method
        self._workload = (rate for rate in workload)
        self._counter = 0
        self.kwargs = kwargs

    def start(self):
        total_seconds = 0
        for rate in self._workload:
            total_seconds += 1
            self._counter += rate
            generator_process = Process(
                # target=self.target_process, args=(rate, queue, ))
                target=self.target_process, args=(rate, total_seconds, ))
            generator_process.daemon = True
            generator_process.start()
            active_children()
            time.sleep(1)
        print(f"{len(active_children())}=")
        print("Spawned all the processes. Waiting to finish...")
        for p in active_children():
            p.join()
        
        print(f"total seconds: {total_seconds}")

        return self._counter, total_seconds

    # def target_process(self, count, queue):
    def target_process(self, count, second):
        asyncio.run(self.generate_load_for_second(count, ))
        responses = asyncio.run(self.generate_load_for_second(count, ))
        print('-'*50)
        print(f"{second=}")
        for response in responses:
            print(response)
        # for response in responses:
        #     queue.put(response)
        print()

    async def generate_load_for_second(self, count):
        # timeout = ClientTimeout(total=TIMEOUT)
        # async with ClientSession(timeout=timeout) as session:
        async with ClientSession() as session:
            delays = np.cumsum(np.random.exponential(
                1 / (count * 1.5), count))
            tasks = []
            for i in range(count):
                task = asyncio.ensure_future(self.predict(delays[i], session))
                tasks.append(task)
            return await asyncio.gather(*tasks)
    
    async def predict(self, delay, session):
        await asyncio.sleep(delay)
        data_id, data = self.get_request_data()
        async with getattr(
            session, self.http_method)(
                self.endpoint, data=data) as response:
            # print('-'*25, 'request sent!', '-'*25)
            response = await response.json(content_type=None)
            # lock.acquire()
            # queue.put(response)
            # lock.release()
            # print('')
            # self.process_response(data_id, response)
            return response
    
    def get_request_data(self) -> Tuple[str, str]:
        return None, None
    
    def process_response(self, data_id: str, response: dict):
        pass

    # def get_responses(self):
    #     outputs = [queue.get() for _ in range(queue.qsize())]
    #     return outputs

# ============= Pure Async Rest based load tester =============


async def request_after_rest(session, url, wait, payload):
    if wait:
        await asyncio.sleep(wait)
    sending_time = time.time()
    try:
        async with session.post(
            url, data=payload, timeout=TIMEOUT) as resp:
            if resp.status != 200:
                inference_response = {'failed': await resp.text()}
            else:
                inference_response = await resp.json()
            resp = {}
            arrival_time = time.time()
            data = inference_response['outputs'][0]['data']
            outputs = {'data': data}
            model_times = eval(inference_response['parameters']['times'])
            times = {
                'request':{
                    'sending': sending_time,
                    'arrival': arrival_time
                }
            }
        times.update(
            {'models': model_times})
        resp['model_name'] = inference_response['model_name']
        resp['outputs'] = [outputs]
        resp['times'] = times
        return resp
    except asyncio.exceptions.TimeoutError:
        resp =  {'failed': 'timeout'}
        arrival_time = time.time()
        times = {
            'request':{
                'sending': sending_time,
                'arrival': arrival_time
            }
        }
        resp.update(times)
        return resp


class BarAzmoonAsyncRest:
    def __init__(self, endpoint, payload, mode, benchmark_duration=1):
        """
        endpoint:
            the http path the load testing endpoint
        payload:
            data to the be sent
        """
        self.endpoint = endpoint
        self.payload = payload
        self.session = aiohttp.ClientSession()
        self.responses = []
        self.duration = benchmark_duration
        self.mode = mode

    async def benchmark(self, request_counts):
        tasks = []
        for i, req_count in enumerate(request_counts):
            tasks.append(
                asyncio.ensure_future(
                    self.submit_requests_after(
                        i * self.duration, req_count, self.duration)
                ))
        await asyncio.gather(*tasks)

    async def submit_requests_after(self, after, req_count, duration):
        if after:
            await asyncio.sleep(after)
        tasks = []
        beta = duration / req_count
        start = time.time()

        rng = default_rng()
        if self.mode == 'step':
            arrival = np.zeros(req_count)
        elif self.mode == 'equal':
            arrival = np.arange(req_count) * beta
        elif self.mode == 'exponential':
            arrival = rng.exponential(beta, req_count)
        print(f'Sending {req_count} requests sent in {time.ctime()} at timestep {after}')
        for i in range(req_count):
            tasks.append(asyncio.ensure_future(
                request_after_rest(
                    self.session,
                    self.endpoint,
                    wait=arrival[i],
                    payload=self.payload
                )
            ))
        resps = await asyncio.gather(*tasks)

        elapsed = time.time() - start
        if elapsed < duration:
            await asyncio.sleep(duration-elapsed)

        self.responses.append(resps)
        print(f'Recieving {len(resps)} requests sent in {time.ctime()} at timestep {after}')

    async def close(self):
        await self.session.close()


# ============= Pure Async Grpc based load tester =============

async def request_after_grpc(stub, metadata, wait, payload):
    if wait:
        await asyncio.sleep(wait)
    sending_time = time.time()
    try:
        # making the strucure similar to the rest output
        resp = {}
        grpc_resp = await stub.ModelInfer(
            request=payload,
            metadata=metadata)
        arrival_time = time.time()
        inference_response = \
            converters.ModelInferResponseConverter.to_types(grpc_resp)
        data = StringRequestCodec.decode_response(inference_response)
        outputs = {'data': data}
        model_times = eval(inference_response.parameters.times)
        times = {
            'request':{
                'sending': sending_time,
                'arrival': arrival_time
            }
        }
        times.update(
            {'models': model_times})
        resp['model_name'] = grpc_resp.model_name
        resp['outputs'] = [outputs]
        resp['times'] = times
        return resp
    except asyncio.exceptions.TimeoutError:
        resp =  {'failed': 'timeout'}
        arrival_time = time.time()
        times = {
            'request':{
                'sending': sending_time,
                'arrival': arrival_time
            }
        }
        resp.update(times)
        return resp
    except grpc.RpcError as e:
        resp =  {'failed': str(e)} 
        arrival_time = time.time()
        times = {
            'request':{
                'sending': sending_time,
                'arrival': arrival_time
            }
        }
        resp.update(times)
        return resp

class BarAzmoonAsyncGrpc:
    # TODO
    def __init__(self, endpoint, metadata, payload, mode, benchmark_duration=1):
        """
        endpoint:
            the path the load testing endpoint
        payload:
            data to the be sent
        """
        self.endpoint = endpoint
        self.payload = payload
        self.metadata = metadata
        self.responses = []
        self.mode = mode
        self.duration = benchmark_duration

    async def benchmark(self, request_counts):
        async with grpc.aio.insecure_channel(self.endpoint) as ch:
            self.stub = dataplane.GRPCInferenceServiceStub(ch)
            tasks = []
            for i, req_count in enumerate(request_counts):
                tasks.append(
                    asyncio.ensure_future(
                        self.submit_requests_after(
                            i * self.duration, req_count, self.duration)
                    ))
            await asyncio.gather(*tasks)

    async def submit_requests_after(self, after, req_count, duration):
        if after:
            await asyncio.sleep(after)
        tasks = []
        beta = duration / req_count
        start = time.time()

        rng = default_rng()
        if self.mode == 'step':
            arrival = np.zeros(req_count)
        elif self.mode == 'equal':
            arrival = np.arange(req_count) * beta
        elif self.mode == 'exponential':
            arrival = rng.exponential(beta, req_count)
        print(f'Sending {req_count} requests sent in {time.ctime()} at timestep {after}')
        for i in range(req_count):
            tasks.append(asyncio.ensure_future(
                request_after_grpc(
                    self.stub,
                    self.metadata,
                    wait=arrival[i],
                    payload=self.payload
                )
            ))
        resps = await asyncio.gather(*tasks)

        elapsed = time.time() - start
        if elapsed < duration:
            await asyncio.sleep(duration-elapsed)

        self.responses.append(resps)
        print(f'Recieving {len(resps)} requests sent in {time.ctime()} at timestep {after}')
