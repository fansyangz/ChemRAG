import torch
import json
import os, sys
def check_env():
    current_env = "/ai/DL/zz"
    r = os.system(f"cd {current_env}")
    current_env = current_env if r == 0 else "/data/zhouyangfan/second"
    sys.path.append(current_env)
    return current_env

current_env = check_env()
from embedding_datasets.milvus.create_collection import get_client, collection_name
from util import batch_func, list_unique
from my_metric.metric import regular_json_str

# current_env = "/data/zhouyangfan/second"

pretrained_model_llama3_embedding = f"{current_env}/llama3_embedding/llama3_embedding.bin"
pretrained_model_tokenizer = f"{current_env}/llama3_embedding/tokenizer.bin"
data_addr = f"{current_env}/embedding_datasets/data1.txt"
config_file_name = f"{current_env}/embedding_datasets/config.json"

tokenizer = torch.load(pretrained_model_tokenizer)

embedding_modules = torch.load(pretrained_model_llama3_embedding).to("cuda")

def get_post_data(sub_word_ids: torch.Tensor, sub_word_emb: torch.Tensor, origin_text, cmpdname):
    # return '''
    # {
    #   "average_embedding": ''' + str(torch.mean(sub_word_emb, dim=0).tolist()) + ''',
    #   "subword_embeddings": ''' + get_sub_word_json(sub_word_ids, sub_word_emb) + ''',
    #   "origin_text": ''' + origin_text + ''',
    #   "cmpdname": ''' + cmpdname + '''
    # }
    # '''
    return {
        "average_embedding": torch.mean(sub_word_emb, dim=0).tolist(),
        # "subword_embeddings": get_sub_word_json(sub_word_ids, sub_word_emb),
        "origin_text": origin_text,
        "cmpdname": cmpdname
    }


def get_sub_word_json(sub_word_ids: torch.Tensor, sub_word_emb: torch.Tensor):
    json_list = {}
    sub_word_ids = sub_word_ids.tolist()
    sub_word_emb = sub_word_emb.tolist()
    for ids, emb in zip(sub_word_ids, sub_word_emb):
        if ids == 128000:
            continue
        json_list[tokenizer.convert_ids_to_tokens(ids)] = emb
    return json_list


def from_word_to_embeddings(word, cmpdname="", return_json=True):
    token_ids = tokenizer(word, return_token_type_ids=False, return_attention_mask=False).input_ids
    sub_word_ids = torch.tensor(token_ids, device="cuda")
    word_vector = embedding_modules(sub_word_ids)
    if return_json:
        return get_post_data(sub_word_ids, word_vector, word, cmpdname)
    else:
        return torch.mean(word_vector, dim=0)


def read_source(*file_path):
    entity_name_result = []
    for f in file_path:
        total_list = []
        with open(f, "r") as file:
            lines = file.readlines()
            for line in lines:
                total_list.append(line.strip())
        total_json_list = [json.loads(str_json) for str_json in total_list]
        for json_result in total_json_list:
            cmpdsynonym = json_result['cmpdsynonym']
            cmpdname = json_result['cmpdname']
            iupacname = json_result['iupacname']
            if cmpdname not in cmpdsynonym:
                cmpdsynonym.append(cmpdname)
            if iupacname not in cmpdsynonym:
                cmpdsynonym.append(iupacname)
            entity_name_result.append({"name_list": cmpdsynonym, "cmpdname": cmpdname})
    total_name_list = []
    for entity_name_json in entity_name_result:
        for name in entity_name_json["name_list"]:
            total_name_list.append(name)
    total_name_count = len(total_name_list)
    return entity_name_result, total_name_count, total_name_list



def do_push_es_vector():
    client = get_client()
    entity_name_result, total_name_count, _ = read_source(data_addr)
    with open(config_file_name, "r") as config_file:
        config = json.load(config_file)
    has_done_start = config["next"]
    counts_per_request = config["per_count"]
    stop_index = config["stop"]
    cache_data = []
    now_index_count = 0
    now_save_count = 0
    start_index = 1
    try_count = 10
    for entity_name_json in entity_name_result:
        name_list = entity_name_json["name_list"]
        cmpdname = entity_name_json["cmpdname"]
        for name in name_list:
            if start_index >= stop_index:
                raise Exception
            if start_index < has_done_start:
                now_index_count = now_index_count + 1
                start_index = start_index + 1
                now_save_count = now_save_count + 1
                continue
            post_data = from_word_to_embeddings(name, cmpdname)
            # es.index(index=index_name, document=post_data)
            cache_data.append(post_data)
            now_index_count = now_index_count + 1
            if len(cache_data) == counts_per_request:
                actions = []
                for i, doc in enumerate(cache_data):
                    # actions.append({"_index": index_name, "_id": start_index, "_source": doc})
                    actions.append({"average_embedding": doc["average_embedding"], "cmpdname": cmpdname, "origin_text": name})
                    start_index = start_index + 1
                print("------------开始保存------------")
                while True:
                    try:
                        client.insert(collection_name=collection_name, data=actions)
                        break
                    except Exception:
                        if try_count == 0:
                            print("------------重试结束------------")
                            raise ValueError
                        try_count = try_count - 1
                        print("------------发生错误，开始重试------------")
                print("------------保存成功------------")
                print(
                    f"------------检索个数:{now_index_count}, 存储个数:{now_save_count}, 总共个数:{total_name_count}------------")
                cache_data.clear()
                now_save_count = now_save_count + counts_per_request
                with open(config_file_name, "w") as file:
                    file.write(json.dumps({"next": start_index, "per_count": counts_per_request, "stop": stop_index}))


def do_push_word_list(word_list):
    client = get_client()
    batch_list = list(batch_func(word_list, batch_size=1000))
    for batch in batch_list:
        batch_insert_data = [{"average_embedding": from_word_to_embeddings(word=word, return_json=False).tolist(),
                              "origin_text": word, "cmpdname": "train_datasets"} for word in batch]
        client.insert(collection_name=collection_name, data=batch_insert_data)
    print("ok!!!!")


def do_push_word_by_file(path=f"{current_env}/data/PolymerAbstracts/train.json"):
    from tuning.llama3_tune_source import data_process_ie
    answer_json_str_list = [t[2]["content"] for t in data_process_ie(path, has_answer=True)]
    entity_list = [entity for entity_list in
                   [json.loads(regular_json_str(answer_json_str))["chemical_compounds"] for
                    answer_json_str in answer_json_str_list] for entity in entity_list]
    do_push_word_list(list_unique(entity_list))


if __name__ == '__main__':
    do_push_word_by_file()
