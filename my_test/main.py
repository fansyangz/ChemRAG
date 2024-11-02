from chatglm.modeling_chatglm import ChatGLMForConditionalGeneration
from chatglm.configuration_chatglm import ChatGLMConfig
from transformers import AutoTokenizer
import torch
from concurrent.futures import ThreadPoolExecutor
import threading
import time

checkpoint = "../chatglm_checkpoint0/model_bin.pth"
tokenizer_ = "/ai/DL/zz/chatglm"
local_config = ChatGLMConfig.from_pretrained(pretrained_model_name_or_path="../chatglm/config.json")
model = ChatGLMForConditionalGeneration(config=local_config, device="cuda")
model.load_state_dict(torch.load(checkpoint), strict=True)
history_init = []


def func():
    thread_name = threading.current_thread().name
    print(f"-----------{thread_name} start-----------")
    start_time = int(time.time())
    response, history = model.chat(AutoTokenizer.from_pretrained(tokenizer_, trust_remote_code=True),
                                   "西南科大是什么", history=history_init)
    used = int(time.time()) - start_time
    print(f"-----------{thread_name} end--time: {used}s---response: {response}")


if __name__ == '__main__':
    thread_num = 1
    threadPool = ThreadPoolExecutor(max_workers=thread_num)
    for i in range(thread_num):
        threadPool.submit(func)
