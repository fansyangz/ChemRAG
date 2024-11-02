import os, sys, argparse
def check_env():
    current_env = "/ai/DL/zz"
    r = os.system(f"cd {current_env}")
    current_env = current_env if r == 0 else "/data/zhouyangfan/second"
    sys.path.append(current_env)
    return current_env

current_env = check_env()
from tuning.llama3_tune_source import data_process_ie, peft_fine_tune, load_lora_model, peft_ppo_mixture_ft, ppo_lora_ft
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

if __name__ == '__main__':
    from my_test.retrieve_test import (train_answer_list_path,
                                       train_no_answer_list_path, answer_list_path, no_answer_list_path, train_answer)
    from util import load_by_name
    # data = {"train_na": load_by_name(train_no_answer_list_path),
    #         "train_a": load_by_name(train_answer_list_path),
    #         "test_na": load_by_name(no_answer_list_path),
    #         "test_a": load_by_name(answer_list_path)}
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group("train_args")
    group.add_argument("--batch_size", type=int, default=4)
    group.add_argument("--local_rank", type=int, default=0)
    group.add_argument("--max_steps", type=int, default=20000)
    group.add_argument("--epoch", type=int, default=200)
    group.add_argument("--use_rag", type=int, default=0)
    group.add_argument("--validate", type=int, default=0)
    group.add_argument("--max_seq_length", type=int, default=2048)
    group.add_argument("--validate_iter", type=int, default=200)
    group.add_argument("--dataset_text_field", type=str, default="formatted_chat")
    group.add_argument("--lora_output", type=str, default="ie-rag-ft")
    group.add_argument("--deepspeed", type=str, default=None)
    args = parser.parse_args()
    args.use_rag = 0
    args.lora_output = "ie-ft-gc"
    data = load_by_name(train_answer_list_path if args.use_rag == 1 else train_answer)
    # _, tokenizer = load_lora_model(only_tokenizer=True)
    # peft_fine_tune(args, data)
    peft_fine_tune(args, data)