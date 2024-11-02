current_env = "/ai/DL/zz"
pretrained_model_llama3 = f"{current_env}/llama3"
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from transformers import (
    AutoTokenizer,
    pipeline,
)


tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model_llama3,
                                          trust_remote_code=True)
llama_pipeline = pipeline("text-generation", model=pretrained_model_llama3, tokenizer=tokenizer, device="cuda")


def func():
    thread_name = threading.current_thread().name
    print(f"-----------{thread_name} start-----------")
    start_time = int(time.time())
    sentences = llama_pipeline("what is the Southwest university of science and technology", do_sample=True, top_k=10, eos_token_id=tokenizer.eos_token_id, max_length=128)
    result = ""
    for seq in sentences:
        result = result + seq["generated_text"]
    used = int(time.time()) - start_time
    print(f"-----------{thread_name} end--time: {used}s---response: {result}")


if __name__ == '__main__':
    thread_num = 5
    threadPool = ThreadPoolExecutor(max_workers=thread_num)
    for i in range(thread_num):
        threadPool.submit(func)
