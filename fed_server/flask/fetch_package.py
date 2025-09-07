import requests
import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))  # RL 根目录
from global_hyparm import *
 
 
url = "http://192.168.68.237:5000"

esp32_ota_package_url = url+"/esp32_ota_package.json"

def fetch_ota_package(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        print("OTA Package Loaded:")
        print(json.dumps(data, indent=4))
        return data
    except Exception as e:
        print("Error fetching OTA package:", e)
        return None

if __name__ == "__main__":
    ota_package = fetch_ota_package(esp32_ota_package_url)
