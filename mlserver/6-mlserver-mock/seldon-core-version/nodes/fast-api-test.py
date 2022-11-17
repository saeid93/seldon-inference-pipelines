# start commnad:
# python -m uvicorn fast-api-test:app
import time
from fastapi import FastAPI
import asyncio

app = FastAPI()
@app.post('/')
async def sumer():
    arrival = time.time()
    await asyncio.sleep(1)
    serving = time.time()
    # to make it consistent with the Seldon data model
    output = {'model_name': 'fastapi', 'outputs': [{'data': [f'{{"time": {{"arrival_fastapi": {arrival}, "serving_fastapi": {serving}}}, "output": []}}']}]}
    return output