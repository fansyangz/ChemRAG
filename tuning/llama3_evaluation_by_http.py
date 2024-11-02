from tuning.llama3_tune_source import data_process_ie, infer_llama_lora_by_prompt, current_env
from tuning.config import flask_http
file_path = f"{current_env}/data/PolymerAbstracts/test.json"

entity_qa_list = data_process_ie(file_path, use_rag=True)
infer_llama_lora_by_prompt(entity_qa_list, write_path="result/result_rag.txt", url=flask_http)