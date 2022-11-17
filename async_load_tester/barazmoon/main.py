import time
from typing import List, Tuple
import numpy as np
from multiprocessing import Process, active_children #, Lock
import asyncio
from aiohttp import ClientSession #, ClientTimeout
from numpy.random import default_rng
import aiohttp
import asyncio
# from multiprocessing import Queue


# MAX_QUEUE_SIZE = 1000000
# TIMEOUT = 20 * 60
# queue = Queue(MAX_QUEUE_SIZE)
# lock = Lock()

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

# ============= Pure Async based load tester =============


async def request_after(session, url, wait, payload):
    if wait:
        await asyncio.sleep(wait)
    sending_time = time.time()
    try:
        async with session.post(url, data=payload) as resp:
            if resp.status != 200:
                resp = {'failed': await resp.text()}  # TODO: maybe raise!
            else:
                resp = await resp.json()
            arrival_time = time.time()
            timing = {
                'timing':{
                    'sending_time': sending_time,
                    'arrival_time': arrival_time
                }
            }
            resp.update(timing)
            return resp
    except asyncio.exceptions.TimeoutError:
        return {'failed': 'timeout'}


class BarAzmoonAsync:
    def __init__(self, endpoint, payload, benchmark_duration=1):
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
        arrival = rng.exponential(beta, req_count)

        print(f'Sending {req_count} requests sent in {time.ctime()} at timestep {after}')
        for i in range(req_count):
            tasks.append(asyncio.ensure_future(
                request_after(
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
