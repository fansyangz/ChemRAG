import requests
import sys, os
from util import list_unique


def check_env():
    current_env = "/ai/DL/zz"
    r = os.system(f"cd {current_env}")
    current_env = current_env if r == 0 else "/data/zhouyangfan/second"
    sys.path.append(current_env)
    return current_env


current_env = check_env()
try:
    from datasets import Dataset
except Exception:
    os.system(f"pip install -r {current_env}/requirements.txt")
    from datasets import Dataset
import time
import torch
import json
from embedding_datasets.milvus.insert_vector import from_word_to_embeddings
from embedding_datasets.milvus.search_vector import search_by_bulk_vector
from tuning.evaluate import EvaluateCallable
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from peft import get_peft_model
from tqdm import tqdm
from torch.utils.data import DataLoader
from accelerate.utils import DeepSpeedPlugin
from jsonformer import Jsonformer

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)
from peft import LoraConfig, PeftModel, LoraModel
from trl import SFTTrainer, SFTConfig

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
pretrained_model_llama3 = f"{current_env}/llama3"
# polymer_datasets = f"{current_env}/data/PolymerAbstracts/train.json"
polymer_datasets = f"{current_env}/data/PolymerAbstracts/test.json"
# lora_output = f"{current_env}/lora-tuned/information-extraction"
material_entities = ['ORGANIC', 'POLYMER', 'INORGANIC', 'POLYMER_FAMILY', 'MONOMER']
property_name = "PROP_NAME"
property_value = "PROP_VALUE"
other = "O"
amount = 'MATERIAL_AMOUNT'
tuning_data = f"{current_env}/data/finetune/ruozhiba_qa.json"
# tuning_output_filename = f"{current_env}/data/finetune/ruozhiba_qa_process.json"
# tuning_output_filename = f"{current_env}/data/finetune/contain_entity_qa.json"
tuning_output_filename = f"{current_env}/data/finetune/information_extraction_prompt.json"
embedding_model = f"{current_env}/llama3_embedding/llama3_embedding.bin"
tokenizer_model = f"{current_env}/llama3_embedding/tokenizer.bin"
# lora_output = f"{current_env}/Llama3-8b-ruozhiba"
sep_token_id = 128002
from accelerate import Accelerator

max_length = 1024
max_steps = 1000


def process_data():
    result_list = []
    with open(tuning_output_filename, "w") as write_file:
        with open(tuning_data, "r") as read_file:
            data_json = json.load(read_file)
            for each_json in data_json:
                each_result_json = {
                    "text": "<s>[INST]" + each_json["instruction"] + "[/INST]" + each_json["output"] + "</s>"}
                result_list.append(each_result_json)
            write_file.write(json.dumps(result_list, ensure_ascii=False, indent=4))


def origin_use(prompt):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model_llama3,
                                              trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=pretrained_model_llama3).to(DEVICE)

    # prompt = "计算机工程与应用怎么样"
    inputs = tokenizer([prompt], max_length=max_length)
    input_ids = torch.tensor(inputs["input_ids"]).to(DEVICE)
    print(input_ids)

    outputs = model.generate(input_ids, max_length=max_length)
    final_result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(final_result)


def pipeline_origin_use(prompt):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model_llama3,
                                              trust_remote_code=True)
    llama_pipeline = pipeline("text-generation", model=pretrained_model_llama3, tokenizer=tokenizer, device=DEVICE)
    sentences = llama_pipeline(prompt, do_sample=True, top_k=10, eos_token_id=tokenizer.eos_token_id,
                               max_length=max_length)
    for seq in sentences:
        print(seq["generated_text"])


def pipeline_origin_use2(prompt):
    model, tokenizer = load_lora_model(train=False)
    llama_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=DEVICE)
    sentences = llama_pipeline(prompt, do_sample=True, top_k=10, eos_token_id=tokenizer.eos_token_id,
                               max_length=max_length)
    for seq in sentences:
        print(seq["generated_text"])


def ie_prompt(document, task):
    return f"<document>{document}</document><task>{task}</task>"


def ie_answer(answer):
    return f"<extraction>{answer}</extraction>"


def ie_prompt_task_demo_json():
    return "{'chemical_compounds': [], 'property_name': [], 'property_value': []}"


def ie_prompt_task_demo():
    return f"from the document, extract the text values and tags of the following entities: " + ie_prompt_task_demo_json()


def get_system_json(use_rag=False):
    content = "extract the text values and tags of the following entities from the document based on the retrieved similar entities during the search: " \
        if use_rag else "from the document, extract the text values and tags of the following entities: "
    return {"role": "system", "content": content + ie_prompt_task_demo_json()}


def get_input_json(document):
    return {"role": "document", "content": document}
    # return {"role": "user", "content": document}


def get_rag_json(document_list, interval_token, word_window=1, topk=10):
    if word_window > 1:
        origin_len = len(document_list)
        for ww in range(1, word_window):
            for i in range(origin_len - ww):
                document_list.append(" ".join(document_list[i:ww + 1 + i]))
    embed_list = [from_word_to_embeddings(d, return_json=False).tolist() for d in document_list]
    retrieve_list = search_by_bulk_vector(embed_list, topk=topk)
    retrieve_word = list_unique([i["origin_text"] for i in retrieve_list])
    return {"role": "retrieve", "content": interval_token.join(retrieve_word)}


def get_assistant_json(answer):
    return {"role": "assistant", "content": answer}


def get_data_json_array(document, answer, no_answer_list=None):
    if no_answer_list:
        no_answer_list.append(get_assistant_json(answer))
        return no_answer_list
    return [get_system_json(), get_input_json(document), get_assistant_json(answer)]


def get_rag_data_json_array(document_list, interval_token, answer, no_answer_list=None, word_window=1, topk=10):
    if no_answer_list:
        no_answer_list.append(get_assistant_json(answer))
        return no_answer_list
    return [get_system_json(True), get_input_json(' '.join(document_list)),
            get_rag_json(document_list, interval_token, word_window=word_window, topk=topk),
            get_assistant_json(answer)]


def get_rag_data_json_array_no_answer(document_list, interval_token, word_window=1, topk=10):
    return [get_system_json(True), get_input_json(' '.join(document_list)),
            get_rag_json(document_list, interval_token, word_window=word_window, topk=topk)]


def get_data_json_array_no_answer(document):
    return [get_system_json(), get_input_json(document)]


def assemble_entity_words(entity_result, last_flag, entity_words):
    tags = ""
    if last_flag == 0:
        tags = "chemical_compounds"
    elif last_flag == 1:
        tags = "property_name"
    elif last_flag == 2:
        tags = "property_value"
    entity_result[tags].append(" ".join(entity_words))
    entity_words.clear()


def ie_prompt_extraction(s_l, n_l):
    last_flag = -1
    entity_words = []
    entity_result = {
        'chemical_compounds': [],
        'property_name': [],
        'property_value': []
    }
    for s, n in zip(s_l, n_l):
        if n == other or n == amount:
            entity_flag = -1
            if last_flag == entity_flag:
                continue
            else:
                assemble_entity_words(entity_result, last_flag, entity_words)
        elif n in material_entities:
            entity_flag = 0
            if last_flag == entity_flag or last_flag == -1:
                entity_words.append(s)
            else:
                assemble_entity_words(entity_result, last_flag, entity_words)
        elif n == property_name:
            entity_flag = 1
            if last_flag == entity_flag or last_flag == -1:
                entity_words.append(s)
            else:
                assemble_entity_words(entity_result, last_flag, entity_words)
        elif n == property_value:
            entity_flag = 2
            if last_flag == entity_flag or last_flag == -1:
                entity_words.append(s)
            else:
                assemble_entity_words(entity_result, last_flag, entity_words)
        else:
            raise ValueError
        last_flag = entity_flag
    return entity_result


def data_process_ie(file_path="../data/PolymerAbstracts/train.json", use_rag=False, has_answer=False,
                    interval_token="|", just_times=0, no_answer_list=None, word_window=1, topk=10):
    total_list = []
    with open(file_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            total_list.append(line.strip())
    total_json_list = [json.loads(str_json) for str_json in total_list]
    sentence_list = [sentence_json["words"] for sentence_json in total_json_list]
    ner_list = [sentence_json["ner"] for sentence_json in total_json_list]
    entity_qa_list = []
    answer_func = lambda x, y: json.dumps(ie_prompt_extraction(x, y)).replace("\"", "'")
    times = 0
    first = True
    for index, (s_l, n_l) in enumerate(zip(sentence_list, ner_list)):
        if 0 < just_times <= times and not first:
            break
        if use_rag:
            if has_answer:
                entity_qa_list.append(get_rag_data_json_array(s_l, interval_token, answer_func(s_l, n_l), topk=topk,
                                                              no_answer_list=no_answer_list[index] if no_answer_list
                                                              else no_answer_list, word_window=word_window))
            else:
                entity_qa_list.append(
                    get_rag_data_json_array_no_answer(s_l, interval_token, word_window=word_window, topk=topk))
        else:
            if has_answer:
                entity_qa_list.append(get_data_json_array(" ".join(s_l), answer_func(s_l, n_l),
                                                          no_answer_list=no_answer_list[index] if no_answer_list
                                                          else no_answer_list))
            else:
                entity_qa_list.append(get_data_json_array_no_answer(" ".join(s_l)))
        first = False
        times = times + 1
    return entity_qa_list


def prepare_base_model_dataset(data, args, train=False, tokenizer=None):
    print("--------------load llm to gpu--------------")
    base_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=pretrained_model_llama3)
    print("--------------load llm to gpu success!--------------")
    base_model.config.use_cache = False
    base_model.config.pretraining_tp = 1
    print("--------------0--------------")
    if not tokenizer:
        _, tokenizer = load_lora_model(only_tokenizer=True, train=train)
    dataset = Dataset.from_dict({"data": data})
    dataset = dataset.map(lambda x: {
        args.dataset_text_field: tokenizer.apply_chat_template(x["data"], tokenize=False, add_generation_prompt=False)})
    return base_model, tokenizer, dataset


def peft_ppo_mixture_ft(args, data, tokenizer=None):
    base_model, tokenizer, dataset = prepare_base_model_dataset(data, args, tokenizer)
    peft_param = LoraConfig(
        lora_alpha=16,  # Lora超参数，用于缩放低秩适应的权重
        lora_dropout=0.1,  # Lora层的丢弃率
        r=64,  # Lora中的秩
        bias="none",
        task_type="CAUSAL_LM"  # Llama属于因果语言模型
    )
    print("--------------2--------------")
    lora_output = f"{current_env}/lora-tuned/{args.lora_output}"
    training_params = SFTConfig(
        output_dir=lora_output,
        num_train_epochs=1,
        gradient_checkpointing=True,  # 模型支持梯度检查点
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_32bit",  # 优化器
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,  # 不适用混合精度训练
        bf16=False,
        max_grad_norm=0.3,  # 裁剪梯度
        warmup_ratio=0.03,
        group_by_length=True,  # 将训练数据集中大致相同长度的样本分分组到同一batch中，提升profill效率
        lr_scheduler_type="constant",  # 学习率调度器将使用常熟衰减策略
        report_to=["tensorboard"],
        per_device_train_batch_size=args.batch_size,
        deepspeed=args.deepspeed,

    )
    lora_model = get_peft_model(base_model, peft_param)
    print("--------------3--------------")
    sft_trainer = SFTTrainer(
        model=lora_model,
        train_dataset=dataset,
        # peft_config=peft_param,
        dataset_text_field=args.dataset_text_field,  # 数据集中用于训练的文本字段
        # formatting_func=formatting_prompts_func,
        # data_collator=collator,
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        args=training_params,
        dataset_batch_size=args.batch_size,
        # packing=False, #不将多个权重参数打包成更少的数据单元进行存储和传输
    )

    ppo_config = PPOConfig(
        learning_rate=1.41e-5,
    )
    ppo_trainer = PPOTrainer(
        model=AutoModelForCausalLMWithValueHead(pretrained_model=base_model),
        config=ppo_config,
        tokenizer=tokenizer,
        dataset=dataset
    )

    def reward_model(outputs):
        from my_metric.metric import do_json_parse
        decoded_output = tokenizer.decode(outputs)
        score = 2.0
        try:
            do_json_parse(decoded_output)
        except Exception:
            score = 0.1
        return torch.FloatTensor(score)

    for epoch in tqdm(range(args.epoch), "total epoch: "):
        for batch in tqdm(ppo_trainer.dataloader):
            sft_trainer.train()
            query_tensors = batch["input_ids"]
            batch_response_tensors = ppo_trainer.generate(query_tensors)
            reward_list = [reward_model(response) for response in batch_response_tensors]
            stats = ppo_trainer.step(query_tensors, [batch_response_tensors], reward_list)
            ppo_trainer.log_stats(stats, batch, reward_list)


class IdsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return torch.tensor(self.dataset[idx])


def ppo_lora_ft(args, data, tokenizer=None):
    # args.deepspeed = f"{current_env}/zero2_config.json"
    lr = 1.41e-5
    _, tokenizer = load_lora_model(only_tokenizer=True, lora_path_simple=args.lora_output, train=False)
    dataset = Dataset.from_dict({"data": data})
    dataset = dataset.map(lambda x: {
        args.dataset_text_field: tokenizer.apply_chat_template(x["data"], tokenize=True, max_length=1024,
                                                               add_generation_prompt=True)})
    ids_dataset = IdsDataset(dataset=dataset[args.dataset_text_field])
    args.batch_size = 1
    dataloader = DataLoader(dataset=ids_dataset, batch_size=args.batch_size)
    ppo_config = PPOConfig(
        learning_rate=lr,
        batch_size=args.batch_size,
        mini_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        # accelerator_kwargs={"deepspeed_plugin": DeepSpeedPlugin(hf_ds_config=args.deepspeed)}
    )
    lora_output = f"{current_env}/lora-tuned/{args.lora_output}"
    model = AutoModelForCausalLMWithValueHead.from_pretrained(lora_output)

    ppo_trainer = PPOTrainer(
        model=model,
        config=ppo_config,
        tokenizer=tokenizer,
        dataset=ids_dataset
    )

    # if args.deepspeed:
    #     accelerator = Accelerator(deepspeed_plugin=DeepSpeedPlugin(hf_ds_config=args.deepspeed))
    #     # optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    #     model, optimizer = accelerator.prepare(model, ppo_trainer.optimizer)

    def reward_model(outputs, input_len):
        from my_metric.metric import do_json_parse
        decoded_output = tokenizer.decode(outputs[input_len:])
        score = 2.0
        try:
            do_json_parse(decoded_output)
        except Exception:
            score = 0.1
        return torch.FloatTensor([score])

    for epoch in tqdm(range(args.epoch), "total epoch: "):
        for batch in tqdm(dataloader):
            input = batch[0].to("cuda")
            input_len = len(input)
            result = ppo_trainer.generate(query_tensor=input,
                                          max_new_tokens=512, eos_token_id=tokenizer.eos_token_id)
            result = result[0]
            reward = reward_model(result, input_len)
            stats = ppo_trainer.step([input], [result], [reward])
            ppo_trainer.log_stats(stats, {"query": input, "response": result}, [reward])


def peft_fine_tune(args, data, tokenizer=None):
    base_model, train_tokenizer, dataset = prepare_base_model_dataset(data=data, args=args, train=True,
                                                                      tokenizer=tokenizer)
    _, test_tokenizer = load_lora_model(train=False, only_tokenizer=True)
    print("--------------1--------------")
    peft_param = LoraConfig(
        lora_alpha=16,  # Lora超参数，用于缩放低秩适应的权重
        lora_dropout=0.1,  # Lora层的丢弃率
        r=64,  # Lora中的秩
        bias="none",
        task_type="CAUSAL_LM"  # Llama属于因果语言模型
    )
    print("--------------2--------------")
    lora_output = f"{current_env}/lora-tuned/{args.lora_output}"
    training_params = SFTConfig(
        output_dir=lora_output,
        num_train_epochs=1000,
        gradient_checkpointing=True,  # 模型支持梯度检查点
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_32bit",  # 优化器
        save_steps=args.validate_iter,
        logging_steps=100,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,  # 不适用混合精度训练
        bf16=False,
        max_grad_norm=0.3,  # 裁剪梯度
        max_steps=args.max_steps,  # 最大训练迭代次数
        warmup_ratio=0.03,
        group_by_length=True,  # 将训练数据集中大致相同长度的样本分分组到同一batch中，提升profill效率
        lr_scheduler_type="constant",  # 学习率调度器将使用常熟衰减策略
        report_to=["tensorboard"],
        per_device_train_batch_size=args.batch_size,
        deepspeed=args.deepspeed,

    )
    print("--------------3--------------")
    trainer = SFTTrainer(
        model=base_model,
        train_dataset=dataset,
        peft_config=peft_param,
        dataset_text_field=args.dataset_text_field,  # 数据集中用于训练的文本字段
        # formatting_func=formatting_prompts_func,
        # data_collator=collator,
        max_seq_length=args.max_seq_length,
        tokenizer=train_tokenizer,
        args=training_params,
        dataset_batch_size=args.batch_size,
        # packing=False, #不将多个权重参数打包成更少的数据单元进行存储和传输
    )
    if args.validate == 1:
        trainer.add_callback(
            EvaluateCallable(trainer=trainer, test_tokenizer=test_tokenizer, validate_iter=args.validate_iter, use_rag=(args.use_rag == 1)))
    print("--------------train start--------------")
    start_time = time.time()
    trainer.train()
    trainer.save_model()
    end_time = time.time()
    print("--------------train end--------------")
    print(f"--------------time: {end_time - start_time}--------------")


def load_peft_model(orgin_model, lora_path=None, lora_path_simple=None):
    if lora_path_simple:
        lora_path = f"{current_env}/lora-tuned/{lora_path_simple}"
    return PeftModel.from_pretrained(orgin_model, lora_path).merge_and_unload()


def load_lora_model(train, only_tokenizer=False, only_model=False, lora_path=None,
                    lora_path_simple=None, can_unmerge=False, base_model=None):
    if lora_path_simple:
        lora_path = f"{current_env}/lora-tuned/{lora_path_simple}"
    merge_model = None
    tokenizer = None
    if not only_tokenizer:
        if not base_model:
            base_model = AutoModelForCausalLM.from_pretrained(pretrained_model_llama3).to("cuda")
        if lora_path:
            merge_model = PeftModel.from_pretrained(base_model, lora_path)
            # 合并原模型和lora微调参数
            if can_unmerge:
                merge_model.merge_adapter()
            else:
                merge_model.merge_and_unload()
        else:
            merge_model = base_model
    if not only_model:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_llama3, trust_remote_code=True)
        tokenizer.sep_token = tokenizer.decode(sep_token_id)
        tokenizer.sep_token_id = sep_token_id
        tokenizer.pad_token = tokenizer.eos_token
        chat_template = "{% for message in messages %}{{bos_token + message['role'] + '\n' + message['content'] + eos_token + '\n'}}{% endfor %}{% if add_generation_prompt %}{{bos_token + 'assistant\n' }}{% endif %}"
        tokenizer.chat_template = chat_template
        # train的时候需要padding在右边，并在句末加入eos，否则模型永远学不会什么时候停下来
        # test的时候需要padding在左边，否则模型生成的结果可能全为eos
        tokenizer.padding_side = "right" if train else "left"
        tokenizer.add_generation_prompt = not train
    return merge_model, tokenizer



def infer_llama_lora_by_prompt_one(prompt_text, model=None, tokenizer=None):
    if not model or not tokenizer:
        model, tokenizer = load_lora_model(train=False)
    llama_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device="cuda")
    inference_results = llama_pipeline(prompt_text, eos_token_id=tokenizer.eos_token_id, max_new_tokens=512)
    return [sentence["generated_text"] for sentence in inference_results]


def infer_llama_lora_by_input_ids(input_ids, lora_path_simple=None, model=None, tokenizer=None):
    model, _ = load_lora_model(train=False, lora_path_simple=lora_path_simple, only_model=True)
    return model.generate(input_ids, eos_token_id=tokenizer.eos_token_id, max_new_tokens=512)


json_schema = {
    "type": "object",
    "properties": {
        "chemical_compounds": {
            "type": "array",
            "items": {"type": "string"}
        },
        "property_name": {
            "type": "array",
            "items": {"type": "string"}
        },
        "property_valuex": {
            "type": "array",
            "items": {"type": "string"}
        }
    }
}


def infer_llama_lora_by_prompt(entity_qa_list, write_path=None, url=None, model=None, tokenizer=None, jsonformer_status=False, lora_path=None):
    if not model or not tokenizer:
        model, tokenizer = load_lora_model(train=False, lora_path_simple=lora_path)
    # 获取推理prompt
    dataset = Dataset.from_dict({"data": entity_qa_list})
    dataset = dataset.map(
        lambda x: {'prompt_text': tokenizer.apply_chat_template(x["data"], tokenize=False, add_generation_prompt=True)})
    # 获得数据集的推理结果
    answer_result_word_list = []
    for index, i in enumerate(dataset):
        print(f"-----------start inference: {index + 1}/ {len(dataset)}------------")
        prompt_text = i["prompt_text"]
        if url:
            result = requests.post(url=url, data={"req": prompt_text}).text
            answer_result_word_list.append(result)
        else:
            if jsonformer_status:
                jsonformer_obj = Jsonformer(model, tokenizer, json_schema, prompt_text)
                inference_results = jsonformer_obj()
                answer_result_word_list.append(inference_results)
            else:
                llama_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device="cuda")
                inference_results = llama_pipeline(prompt_text, eos_token_id=tokenizer.eos_token_id, max_new_tokens=512)
                inner_sentences = [sentence["generated_text"] for sentence in inference_results]
                answer_result_word_list.append(inner_sentences[0])
    if write_path:
        with open(write_path, "w") as file:
            for answer in answer_result_word_list:
                file.write(answer)
                file.write("\n")
    print('ok!')
    return answer_result_word_list
