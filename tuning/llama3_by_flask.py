import sys, os
def check_env():
    current_env = "/ai/DL/zz"
    r = os.system(f"cd {current_env}")
    current_env = current_env if r == 0 else "/data/zhouyangfan/second"
    sys.path.append(current_env)
    return current_env

current_env = check_env()

from flask import Flask, request
app = Flask(__name__)

from tuning.llama3_tune_source import load_lora_model, infer_llama_lora_by_prompt_one
pretrained_model_llama3 = f"{current_env}/llama3"
lora_output = f"{current_env}/lora-tuned/information-extraction"
model, tokenizer = load_lora_model(lora_path=lora_output, train=False)

@app.route("/", methods=["POST"])
def req():
    req = request.form.get("req")
    return infer_llama_lora_by_prompt_one(req, model, tokenizer)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=11000)
