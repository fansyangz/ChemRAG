import sys, os
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
# import torch.distributed as dist
import torch
import json
import argparse
pretrained_model_llama3 = f"{current_env}/llama3"
tuning_data = f"{current_env}/data/finetune/ruozhiba_qa.json"
tuning_output_filename = f"{current_env}/data/finetune/information_extraction_prompt.json"
lora_output = f"{current_env}/lora-tuned/information-extraction"
max_length = 128
max_steps = 1000
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

prompt_special_token = ["<document>", "</document>", "<task>", "</task>", "<extraction>", "</extraction>"]
material_entities = ['ORGANIC', 'POLYMER', 'INORGANIC', 'POLYMER_FAMILY', 'MONOMER']
property_name = "PROP_NAME"
property_value = "PROP_VALUE"
other = "O"
amount = 'MATERIAL_AMOUNT'
task_special = "<task>"
answer_special = "<answer>"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
# dist.init_process_group(backend="nccl")

def process_data():
    result_list = []
    with open(tuning_output_filename, "w") as write_file:
        with open(tuning_data, "r") as read_file:
            data_json = json.load(read_file)
            for each_json in data_json:
                each_result_json = {"text": "<s>[INST]" + each_json["instruction"] + "[/INST]" + each_json["output"] + "</s>"}
                result_list.append(each_result_json)
            write_file.write(json.dumps(result_list, ensure_ascii=False, indent=4))


def origin_use(prompt):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model_llama3, trust_remote_code=True)
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
    sentences = llama_pipeline(prompt, do_sample=True, top_k=10, eos_token_id=tokenizer.eos_token_id, max_length=max_length)
    for seq in sentences:
        print(seq["generated_text"])


def ie_prompt(text, task):
    return f"{task}\n{text}"


def ie_prompt_task_demo_json():
    return "{'chemical_compounds': [], 'property_name': [], 'property_value': []}"


def ie_prompt_task_demo():
    return f"from the text below, extract the text values and tags of the following entities: " + ie_prompt_task_demo_json()

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


def data_process_ie():
    total_list = []
    with open("../data/PolymerAbstracts/train.json", "r") as file:
        lines = file.readlines()
        for line in lines:
            total_list.append(line.strip())
    total_json_list = [json.loads(str_json) for str_json in total_list]
    sentence_list = [sentence_json["words"] for sentence_json in total_json_list]
    ner_list = [sentence_json["ner"] for sentence_json in total_json_list]
    result_list = []
    result_json_list = []
    for s_l, n_l in zip(sentence_list, ner_list):
        sentence_json = ie_prompt_extraction(s_l, n_l)
        json_str = json.dumps(sentence_json)
        json_str = json_str.replace("\"", "'")
        result_json_list.append(get_data_json_array(" ".join(s_l), json_str))
    return result_json_list


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['task'])):
        text = f"{task_special}{example['task'][i]}\n{answer_special}{example['answer'][i]}"
        output_texts.append(text)
    return output_texts


def get_system_json():
    return {"role": "system", "content": f"from the document, extract the text values and tags of the following entities: " + ie_prompt_task_demo_json()}

def get_input_json(document):
    return {"role": "document", "content": document}

def get_assistant_json(answer):
    return {"role": "assistant", "content": answer}


def get_data_json_array(document, answer):
    return [get_system_json(), get_input_json(document), get_assistant_json(answer)]


def peft_fine_tune(args):
    # dataset = load_dataset("json", data_files=tuning_output_filename, split="train")
    print("--------------load llm to gpu--------------")
    base_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=pretrained_model_llama3).to(DEVICE)
    print("--------------load llm to gpu success!--------------")
    base_model.config.use_cache = False
    base_model.config.pretraining_tp = 1
    print("--------------0--------------")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model_llama3, trust_remote_code=True)
    chat_template = "{% for message in messages %}{{bos_token + message['role'] + '\n' + message['content'] + eos_token + '\n'}}{% endfor %}{% if add_generation_prompt %}{{bos_token + 'assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template
    tokenizer.pad_token = tokenizer.eos_token
    dataset = Dataset.from_dict({"data": data_process_ie()})
    dataset = dataset.map(lambda x: {
        args.dataset_text_field: tokenizer.apply_chat_template(x["data"], tokenize=False, add_generation_prompt=False)})
    # for special_token in prompt_special_token:
    #     tokenizer.additional_special_tokens(AddedToken(content=special_token))
    # train的时候需要padding在右边，并在句末加入eos，否则模型永远学不会什么时候停下来
    # test的时候需要padding在左边，否则模型生成的结果可能全为eos
    tokenizer.padding_side = "right"
    print("--------------1--------------")
    peft_param = LoraConfig(
        lora_alpha=16, #Lora超参数，用于缩放低秩适应的权重
        lora_dropout=0.1, #Lora层的丢弃率
        r=64, # Lora中的秩
        bias="none",
        task_type="CAUSAL_LM" #Llama属于因果语言模型
    )
    print("--------------2--------------")
    training_params = SFTConfig(
        output_dir=lora_output,
        num_train_epochs=1000,
        gradient_checkpointing=True, #模型支持梯度检查点
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_32bit", #优化器
        save_steps=args.max_steps / 2,
        logging_steps=100,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False, #不适用混合精度训练
        bf16=False,
        max_grad_norm=0.3, #裁剪梯度
        max_steps=args.max_steps, #最大训练迭代次数
        warmup_ratio=0.03,
        group_by_length=True, # 将训练数据集中大致相同长度的样本分分组到同一batch中，提升profill效率
        lr_scheduler_type="constant", #学习率调度器将使用常熟衰减策略
        report_to=["tensorboard"],
        per_device_train_batch_size=args.batch_size,
        deepspeed=args.deepspeed,
        
    )
    print("--------------3--------------")
    # collator = DataCollatorForCompletionOnlyLM(instruction_template=task_special, response_template=answer_special, tokenizer=tokenizer, mlm=False)
    trainer = SFTTrainer(
        model=base_model,
        train_dataset=dataset,
        peft_config=peft_param,
        dataset_text_field=args.dataset_text_field, #数据集中用于训练的文本字段
        # formatting_func=formatting_prompts_func,
        # data_collator=collator,
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        args=training_params,
        dataset_batch_size=args.batch_size,
        # packing=False, #不将多个权重参数打包成更少的数据单元进行存储和传输
    )

    print("--------------train start--------------")
    start_time = time.time()
    trainer.train()
    trainer.save_model()
    end_time = time.time()
    print("--------------train end--------------")
    print(f"--------------time: {end_time-start_time}--------------")


def infer_llama_lora(instruction):
    base_model = AutoModelForCausalLM.from_pretrained(pretrained_model_llama3)
    merge_model = PeftModel.from_pretrained(base_model, f"{lora_output}/checkpoint-{max_steps}")
    #合并原模型和lora
    merge_model = merge_model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_llama3, trust_remote_code=True)
    # tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.additional_special_tokens(prompt_special_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    llama_pipeline = pipeline("text-generation", model=merge_model, tokenizer=tokenizer)
    sentences = llama_pipeline(f"<s>[INST]{instruction}[/INST]", eos_token_id=tokenizer.eos_token_id, max_new_tokens=256)

    for seq in sentences:
        print(seq["generated_text"])

# peft_fine_tune()
# infer_llama_lora("只剩一个心脏了还能活吗？")
# process_data()
# origin_use("只剩一个心脏了还能活吗？")
# pipeline_origin_use("只剩一个心脏了还能活吗？")
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group("train_args")
    group.add_argument("--batch_size", type=int, default=8)
    group.add_argument("--max_steps", type=int, default=20000)
    group.add_argument("--max_seq_length", type=int, default=2048)
    group.add_argument("--dataset_text_field", type=str, default="formatted_chat")
    group.add_argument("--deepspeed", type=str, default=f"{current_env}/zero3_config.json")
    args = parser.parse_args()
    peft_fine_tune(args)