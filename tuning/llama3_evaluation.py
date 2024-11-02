import os, sys

import torch.cuda


def check_env():
    current_env = "/ai/DL/zz"
    r = os.system(f"cd {current_env}")
    current_env = current_env if r == 0 else "/data/zhouyangfan/second"
    sys.path.append(current_env)
    return current_env

current_env = check_env()
from util import load_by_name


file_path_test = f"{current_env}/data/PolymerAbstracts/test.json"
file_path_train = f"{current_env}/data/PolymerAbstracts/train.json"
file_path_dev = f"{current_env}/data/PolymerAbstracts/dev.json"

word_window = 3
topk = 10
test_topk = 30
write_path_default = lambda x: f"{current_env}/tuning/result_rag_ww{word_window}_topk{topk}_train.txt" if (
        x == file_path_train) else f"{current_env}/tuning/result_rag_ww{word_window}_topk{topk}_test.txt"


def do_evaluation(model=None, tokenizer=None, write_path=None, use_rag=False):
    from tuning.llama3_tune_source import infer_llama_lora_by_prompt
    from my_test.retrieve_test import no_answer_list_path, no_answer
    use_rag = True
    return infer_llama_lora_by_prompt(load_by_name(no_answer_list_path if use_rag else no_answer), write_path=write_path, model=model,
                                      tokenizer=tokenizer, jsonformer_status=True, lora_path="ie-ft-rag/checkpoint-10000")



def do_exist_evaluation():
    from tuning.llama3_tune_source import infer_llama_lora_by_prompt, load_lora_model, load_peft_model
    from my_test.retrieve_test import no_answer_list_path, no_answer
    from metric import metric_by_result
    # origin_model, tokenizer = load_lora_model(train=False, can_unmerge=True)
    use_rag = True
    prompt = load_by_name(no_answer_list_path if use_rag else no_answer)
    merge_model = None
    for index in range(200, 20000, 200):
        if merge_model:
            merge_model.unload()
        merge_model, tokenizer = load_lora_model(train=False, base_model=merge_model,
                                                 lora_path_simple=f"ie-ft-rag/checkpoint-{index}", can_unmerge=True)
        result_json = infer_llama_lora_by_prompt(prompt, model=merge_model, tokenizer=tokenizer,
                                                 jsonformer_status=True)
        print(f"----------index: {index}-----------")
        metric_by_result(result_json, gc=True)


def do_exist_evaluation_once(index):
    from tuning.llama3_tune_source import infer_llama_lora_by_prompt, load_lora_model, load_peft_model
    from my_test.retrieve_test import no_answer_list_path, no_answer
    from metric import metric_by_result
    # origin_model, tokenizer = load_lora_model(train=False, can_unmerge=True)
    use_rag = True
    prompt = load_by_name(no_answer_list_path if use_rag else no_answer)
    merge_model, tokenizer = load_lora_model(train=False, lora_path_simple=f"ie-ft-rag/checkpoint-{index}" if index else None)
    result_json = infer_llama_lora_by_prompt(prompt, model=merge_model, tokenizer=tokenizer,
                                             jsonformer_status=True)
    print(f"----------index: {index}-----------")
    metric_by_result(result_json, gc=True)


if __name__ == '__main__':
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    do_exist_evaluation_once(None)
    print("ok!!!")
