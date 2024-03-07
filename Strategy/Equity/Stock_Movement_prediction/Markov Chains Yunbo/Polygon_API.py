# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:41:59 2023

@author: jrjol
"""
from polygon import RESTClient

client = RESTClient(api_key="ocunxnOqC0pnltRqT3VkOiKeCmPE49L7")

import requests

url = "https://api.polygon.io/v2/aggs/ticker/DELL/range/1/day/2010-10-22/2023-10-23?adjusted=true&sort=asc&limit=5000&apiKey=ocunxnOqC0pnltRqT3VkOiKeCmPE49L7"

response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    print(data)
else:
    print("Failed to retrieve data. Status code:", response.status_code)
import pandas as pd

results = data["results"]
df = pd.DataFrame(results)
df = df[["o", "h", "l", "c"]]

print(results)
