from transformers import AutoTokenizer, AutoModel
from chatglm.modeling_chatglm import ChatGLMForConditionalGeneration
from chatglm.configuration_chatglm import ChatGLMConfig

checkpoint = "/ai/DL/zz/chatglm"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model = ChatGLMForConditionalGeneration(config=ChatGLMConfig.from_pretrained(pretrained_model_name_or_path="../chatglm/config.json"))
history_init = [{"content": "You are currently a chemical expert. Please extract chemical properties and values from the text and directly output their by json", "role":"user"}]
response, history = model.chat(tokenizer, "A novel conjugated polymer, poly(3,4-bisphenyl-N-n-hexyl-pyrrole-2,5-dione), was synthesized by dehalogenation polycondensation with the zerovalent nickel complexes ( the Yamamoto method ) , of which the chemical architecture was based on the polyarylenevinylene having a fixed cis - vinylene group . This polymer showed high solubility in common organic solvents , high spin - castability and high thermal stability up to ca . 400 \u00b0 C and the T_{g } of 152 \u00b0 C . The dilute polymer solution ( THF ) showed the greenish yellow photoluminescence with the maximum at 526 nm .",
                               history=history_init)

print(response)