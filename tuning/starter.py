import os
current_env = "/ai/DL/zz"
import sys
sys.path.append(current_env)
import argparse

WORLD_SIZE = os.getenv("WORLD_SIZE")
RANK = os.getenv("RANK")
MASTER_ADDR = os.getenv("MASTER_ADDR")
MASTER_PORT = os.getenv("MASTER_PORT")

IS_AI_PLATFORM = WORLD_SIZE is not None

parser = argparse.ArgumentParser()
group = parser.add_argument_group("train_args")
group.add_argument("--batch_size", type=int, default=4)
group.add_argument("--zero", type=int, default=3)
group.add_argument("--node", type=int, default=0)
group.add_argument("--max_steps", type=int, default=20000)
group.add_argument("--epoch", type=int, default=200)
group.add_argument("--max_seq_length", type=int, default=2048)
group.add_argument("--validate_iter", type=int, default=500)
group.add_argument("--dataset_text_field", type=str, default="formatted_chat")
group.add_argument("--lora_output", type=str, default="ie-rag/checkpoint-10000")
group.add_argument("--start_file", type=str, default="tuning/llama3_tune.py")
args = parser.parse_args()

DEV_NODES = [
    {
        "host": "222.196.35.22",
        "port": 35520
    },
    {
        "host": "222.196.35.22",
        "port": 36857
    },
]

DEV_NODES_INNER = [
    {
        "host": "162.30.2.39",
        "port": 35520
    },
    {
        "host": "192.168.0.3",
        "port": 36857
    },
]

print("******start training******")
if IS_AI_PLATFORM:
    if args.node == 1:
        print("-----------------start deepspeed mutiGPU-------------------------------")
        os.system(f"deepspeed --num_gpus=2 {current_env}/{args.start_file} --batch_size={args.batch_size} --max_steps={args.max_steps} --max_seq_length={args.max_seq_length} --deepspeed={current_env}/zero{args.zero}_config.json")
    else:
        print("-----------------start torchrun-----------------")
        os.system(f"torchrun --nproc_per_node=1 --nnodes={WORLD_SIZE} --node_rank={RANK} --master_addr={MASTER_ADDR} --master_port={MASTER_PORT} {current_env}/{args.start_file} --batch_size={args.batch_size} --max_steps={args.max_steps} --max_seq_length={args.max_seq_length} --deepspeed={current_env}/zero{args.zero}_config.json")
else:
    hosts = []
    with open(f"{current_env}/hostfile", "r") as file:
        hosts = hosts + file.readlines()
    print("-----------------start deepspeed-----------------")
    os.system(f"deepspeed --num_gpus=1 --hostfile={current_env}/hostfile --num_nodes={len(hosts)} --master_addr={DEV_NODES_INNER[0]['host']} --master_port=9901 {current_env}/{args.start_file} --batch_size={args.batch_size} --max_steps={args.max_steps} --max_seq_length={args.max_seq_length} --deepspeed={current_env}/zero{args.zero}_config.json")
