import requests
import sys, os
def check_env():
    current_env = "/ai/DL/zz"
    r = os.system(f"cd {current_env}")
    current_env = current_env if r == 0 else "/data/zhouyangfan/second"
    sys.path.append(current_env)
    return current_env

current_env = check_env()
from tuning.config import flask_http

result = requests.post(url=flask_http, data={"req": "what is the apple?"}).text

print(result)