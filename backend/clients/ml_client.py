import requests
from config import ML_SERVICE_URL

def call_agent1(payload):
    url = f"{ML_SERVICE_URL}/api/market/data"
    response = requests.post(url, json=payload)
    return response.json()

def call_agent2(payload):
    url = f"{ML_SERVICE_URL}/api/decision/make"
    response = requests.post(url, json=payload)
    return response.json()

def call_agent3(payload):
    url = f"{ML_SERVICE_URL}/api/execution/execute"
    response = requests.post(url, json=payload)
    return response.json()