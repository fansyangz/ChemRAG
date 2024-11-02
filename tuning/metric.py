import re
import json

import os, sys
def check_env():
    current_env = "/ai/DL/zz"
    r = os.system(f"cd {current_env}")
    current_env = current_env if r == 0 else "/data/zhouyangfan/second"
    sys.path.append(current_env)
    return current_env
current_env = check_env()
assistant_start_token = "<|begin_of_text|>assistant"
assistant_response_regex = r"(?<=<\|begin_of_text\|>assistant\n).+"
from tuning.llama3_evaluation import do_evaluation

def fetch_result_by_path(file_path):
    from llama3_evaluation import write_path_default
    result_list = []
    with open(file_path, "r") as file:
        is_answer = False
        for e in file.readlines():
            if e.startswith(assistant_start_token):
                is_answer = True
                continue
            if is_answer:
                result_list.append(e)
                is_answer = False
    return result_list


def fetch_result_by_list(outputs_list):
    result_list = []
    pattern = re.compile(assistant_response_regex)
    for outputs in outputs_list:
        matches = pattern.search(outputs)
        if matches:
            result_list.append(matches.group())
    return result_list


def metric_by_result(result_list, file_path=None, gc=False):
    from llama3_evaluation import file_path_test
    from my_metric.metric import do_metric
    from llama3_tune_source import data_process_ie
    data = data_process_ie(file_path if file_path else file_path_test, has_answer=True)
    data = [e[2]["content"] for e in data]
    precision, recall, f1, statistic_error_json = do_metric(result_list, data, statistic_error_json=True, gc=gc)
    print(f"precision:{precision}, recall:{recall}, f1:{f1}, statistic_error_json:{statistic_error_json}")
    return precision, recall, f1, statistic_error_json

def metric_by_result_with_error_json(result_list):
    from llama3_evaluation import file_path_test
    from my_metric.metric import do_metric
    from llama3_tune_source import data_process_ie
    data = data_process_ie(file_path_test, has_answer=True)
    data = [e[2]["content"] for e in data]
    precision, recall, f1, error_json_ratio = do_metric(result_list, data, statistic_error_json=True)
    print(f"precision:{precision}, recall:{recall}, f1:{f1}, error_json_ratio:{error_json_ratio}")
    return precision, recall, f1, error_json_ratio


def metric_by_outputs(outputs_list):
    return metric_by_result_with_error_json(fetch_result_by_list(outputs_list))


def from_model_to_result(model, tokenizer, write_path=None, use_rag=False):
    metric_by_result(do_evaluation(model, tokenizer, write_path=write_path, use_rag=use_rag))


def write_result():
    result_list = []
    for index in range(200, 20000, 200):
        precision, recall, f1, statistic_error_json = metric_by_result(
            fetch_result_by_path(f"{current_env}/tuning/result/result_lora_rag_checkpoint_{index}.txt"))
        result_list.append({"index": index, "precision": precision, "recall": recall, "f1": f1,
                            "statistic_error_json": statistic_error_json})
    with open(f"{current_env}/tuning/result/lora_rag_result.txt", "w") as file:
        for json_obj in result_list:
            json_str = json.dumps(json_obj) + "\n"
            file.write(json_str)


# if __name__ == '__main__':
    # print(123123)
    # from llama3_tune_source import load_lora_model
    # import argparse
    # parser = argparse.ArgumentParser()
    # group = parser.add_argument_group("train_args")
    # group.add_argument("--checkpoint", type=int)
    # args = parser.parse_args()
    # model, tokenizer = load_lora_model(lora_path_simple=f"ie-ft-rag/checkpoint-{args.checkpoint}", train=False)
    # from_model_to_result(model, tokenizer, use_rag=True,
    #                      write_path=f"{current_env}/tuning/result/result_lora_rag_checkpoint_{args.checkpoint}.txt")



    # list = [f"afdsfaddfasdf{assistant_start_token}\n12312312aaaa3123123"]
    # print(fetch_result_by_list(list))
