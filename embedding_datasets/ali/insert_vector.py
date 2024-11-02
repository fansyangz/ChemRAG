import torch
import json
from create_collection import collection

current_env = "/data/zhouyangfan/second"

pretrained_model_llama3_embedding = f"{current_env}/embedding_datasets/llama3_embedding.bin"
pretrained_model_tokenizer = f"{current_env}/embedding_datasets/tokenizer.bin"
data_addr = f"{current_env}/embedding_datasets/data1.txt"
config_file_name = f"{current_env}/embedding_datasets/config.json"

tokenizer = torch.load(pretrained_model_tokenizer)

embedding_modules = torch.load(pretrained_model_llama3_embedding)

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


def from_word_to_embeddings(word):
    token_ids = tokenizer(word, return_token_type_ids=False, return_attention_mask=False).input_ids
    sub_word_ids = torch.tensor(token_ids, device="cuda")
    word_vector = embedding_modules(sub_word_ids)
    return get_post_data(sub_word_ids, word_vector, "", "")


def do_push_es_vector():
    total_list = []
    with open(data_addr, "r") as file:
        lines = file.readlines()
        for line in lines:
            total_list.append(line.strip())
    total_json_list = [json.loads(str_json) for str_json in total_list]
    entity_name_result = []
    for json_result in total_json_list:
        cmpdsynonym = json_result['cmpdsynonym']
        cmpdname = json_result['cmpdname']
        iupacname = json_result['iupacname']
        if cmpdname not in cmpdsynonym:
            cmpdsynonym.append(cmpdname)
        if iupacname not in cmpdsynonym:
            cmpdsynonym.append(iupacname)
        entity_name_result.append({"name_list": cmpdsynonym, "cmpdname": cmpdname})
    with open(config_file_name, "r") as config_file:
        config = json.load(config_file)

    has_done_start = config["next"]
    counts_per_request = config["per_count"]
    cache_data = []
    total_name_count = sum([len(entity_name_json["name_list"]) for entity_name_json in entity_name_result])
    now_index_count = 0
    now_save_count = 0
    start_index = 1

    try_count = 10

    for entity_name_json in entity_name_result:
        name_list = entity_name_json["name_list"]
        cmpdname = entity_name_json["cmpdname"]
        for name in name_list:
            if start_index < has_done_start:
                now_index_count = now_index_count + 1
                start_index = start_index + 1
                now_save_count = now_save_count + 1
                continue
            post_data = from_word_to_embeddings(name)
            # es.index(index=index_name, document=post_data)
            cache_data.append(post_data)
            now_index_count = now_index_count + 1
            if len(cache_data) == counts_per_request:
                actions = []
                for i, doc in enumerate(cache_data):
                    # actions.append({"_index": index_name, "_id": start_index, "_source": doc})
                    actions.append((str(start_index), doc["average_embedding"], {"cmpdname": cmpdname, "origin_text": doc["origin_text"]}))
                    start_index = start_index + 1
                print("------------开始保存------------")
                while True:
                    try:
                        assert collection.insert(actions)
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
                    file.write(json.dumps({"next": start_index, "per_count": counts_per_request}))


if __name__ == '__main__':
    do_push_es_vector()
