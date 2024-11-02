import sys, os
import pickle
def check_env():
    current_env = "/ai/DL/zz"
    r = os.system(f"cd {current_env}")
    current_env = current_env if r == 0 else "/data/zhouyangfan/second"
    sys.path.append(current_env)
    return current_env
current_env = check_env()

from tuning.llama3_tune_source import data_process_ie, load_lora_model
from tuning.llama3_evaluation import file_path_test, file_path_train,  word_window, topk, test_topk


no_answer_list_path = f"{current_env}/my_test/no_answer_list_ww{word_window}_topk{test_topk}.pkl"
answer_list_path = f"{current_env}/my_test/answer_list_ww{word_window}_topk{test_topk}.pkl"
train_no_answer_list_path = f"{current_env}/my_test/no_answer_list_ww{word_window}_topk{topk}_train.pkl"
train_answer_list_path = f"{current_env}/my_test/answer_list_ww{word_window}_topk{topk}_train.pkl"
train_answer = f"{current_env}/my_test/answer_list_train.pkl"
train_no_answer = f"{current_env}/my_test/no_answer_list_train.pkl"
answer = f"{current_env}/my_test/answer_list.pkl"
no_answer = f"{current_env}/my_test/no_answer_list.pkl"

def save_retrieve_json_obj():
    sep_token = load_lora_model(only_tokenizer=True, train=True)[1].sep_token
    print("----------no_answer_list start---------------")
    no_answer_list = data_process_ie(file_path=file_path_train, use_rag=False, has_answer=False, interval_token=sep_token,
                                     word_window=word_window, topk=topk)
    with open(train_no_answer, "wb") as file:
        pickle.dump(no_answer_list, file)
    print("----------no_answer_list end---------------")
    print("----------answer_list start---------------")
    answer_list = data_process_ie(file_path=file_path_train, use_rag=False, has_answer=True, interval_token=sep_token,
                                  no_answer_list=no_answer_list, word_window=word_window, topk=topk)
    with open(train_answer, "wb") as file:
        pickle.dump(answer_list, file)
    print("----------answer_list end---------------")

    print("----------no_answer_list start---------------")
    no_answer_list = data_process_ie(file_path=file_path_test, use_rag=False, has_answer=False, interval_token=sep_token,
                                     word_window=word_window, topk=topk)
    with open(no_answer, "wb") as file:
        pickle.dump(no_answer_list, file)
    print("----------no_answer_list end---------------")
    print("----------answer_list start---------------")
    answer_list = data_process_ie(file_path=file_path_test, use_rag=False, has_answer=True, interval_token=sep_token,
                                  no_answer_list=no_answer_list, word_window=word_window, topk=topk)
    with open(answer, "wb") as file:
        pickle.dump(answer_list, file)
    print("----------answer_list end---------------")
    print("ok!!!")


def load_retrieve_json_obj():
    with open("no_answer_list_ww3_topk30.pkl", "rb") as file:
        no_answer_list = pickle.load(file)

    with open("answer_list_ww3_topk30.pkl", "rb") as file:
        answer_list = pickle.load(file)
    print(no_answer_list, answer_list)


if __name__ == '__main__':
    # load_retrieve_json_obj()
    save_retrieve_json_obj()





