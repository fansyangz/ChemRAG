import sys, os
def check_env():
    current_env = "/ai/DL/zz"
    r = os.system(f"cd {current_env}")
    current_env = current_env if r == 0 else "/data/zhouyangfan/second"
    sys.path.append(current_env)
    return current_env
current_env = check_env()
from tuning.llama3_tune_source import load_lora_model

_, tokenizer = load_lora_model(only_tokenizer=True, train=False)
result = tokenizer.encode("3-hydroxybutyrate-co-hydroxyvalerate")
print(result)
for i in result:
    print(tokenizer.decode(i))