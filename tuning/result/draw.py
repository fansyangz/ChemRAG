import matplotlib.pyplot as plt
import json
import os, sys

def check_env():
    current_env = "/ai/DL/zz"
    r = os.system(f"cd {current_env}")
    current_env = current_env if r == 0 else "/data/zhouyangfan/second"
    sys.path.append(current_env)
    return current_env
current_env = check_env()

# file_path = f"{current_env}/tuning/result/lora_rag_result.txt"
# file_path = f"{current_env}/tuning/result/lora_result.txt"
file_path = f"{current_env}/tuning/result/lora_gc_result.txt"
# file_path = f"{current_env}/tuning/result/lora_gc_rag_result.txt"
with open(file_path, 'r') as file:
    data = [json.loads(line) for line in file]

# 提取数据
index = [d["index"] for d in data]
f1_scores = [d["f1"] for d in data]
error_json = [d["statistic_error_json"] for d in data]

# 创建画布和子图
fig, ax1 = plt.subplots()

# 绘制左侧y轴对应的折线
color = 'tab:blue'
ax1.set_xlabel('Step')
ax1.set_ylabel('F1', color=color)
ax1.plot(index, f1_scores, color=color, marker='o')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(0.45, 0.80)

# 创建第二个y轴
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Error JSON', color=color)
ax2.plot(index, error_json, color=color, marker='x')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0.0, 1.0)

# 添加标题和显示图形
plt.title('w GC w/o RAG')
# plt.show()
plt.savefig('gc_no_rag.svg', format='svg', bbox_inches='tight')