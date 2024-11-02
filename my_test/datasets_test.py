import sys, os
import json
def check_env():
    current_env = "/ai/DL/zz"
    r = os.system(f"cd {current_env}")
    current_env = current_env if r == 0 else "/data/zhouyangfan/second"
    sys.path.append(current_env)
    return current_env
current_env = check_env()
from tuning.llama3_tune_source import ie_prompt_extraction

file_path = "../data/PolymerAbstracts/test.json"

total_list = []
with open(file_path, "r") as file:
    lines = file.readlines()
    for line in lines:
        total_list.append(line.strip())
total_json_list = [json.loads(str_json) for str_json in total_list]
sentence_list = [sentence_json["words"] for sentence_json in total_json_list]
ner_list = [sentence_json["ner"] for sentence_json in total_json_list]
entity_qa_list = []
answer_func = lambda x, y: ie_prompt_extraction(x, y)

e = 0
s = 0
p = 0
for index, (s_l, n_l) in enumerate(zip(sentence_list, ner_list)):
    json = answer_func(s_l, n_l)
    e = e + len(json["chemical_compounds"])
    s = s + len(json["property_name"])
    p = p + len(json["property_value"])
print("ok")

