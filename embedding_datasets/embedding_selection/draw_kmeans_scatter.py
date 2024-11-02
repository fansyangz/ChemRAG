import torch
from torch.utils.data import DataLoader
import sys, os
import argparse
def check_env():
    current_env = "/ai/DL/zz"
    r = os.system(f"cd {current_env}")
    current_env = current_env if r == 0 else "/data/zhouyangfan/second"
    sys.path.append(current_env)
    return current_env

# current_env = check_env()
from tuning.llama3_tune_source import current_env
from embedding_datasets.milvus.insert_vector import from_word_to_embeddings, read_source
from embedding_datasets.embedding_selection.cluter_model import KMeans, Embedding_Dataset
# torch.cuda.empty_cache()
path1 = f"{current_env}/embedding_datasets/data1.txt"
path2 = f"{current_env}/embedding_datasets/data2.txt"
path3 = f"{current_env}/embedding_datasets/data3.txt"

if __name__ == '__main__':
    model_file_name = "cluster_100000-epoch_100-file_dataall.bin"
    model = torch.load(model_file_name)
    _, total_name_count, total_name_list = read_source(path1, path2, path3)
    print(f"-----------------------total_name_count: {total_name_count}-----------------------")
    print("-----------------------get embedding from file to datasets-----------------------")
    datasets = Embedding_Dataset(name_list=total_name_list)
    print("-----------------------to dataloader-----------------------")
    dataloader = DataLoader(dataset=datasets, batch_size=20000)
    for en, x in enumerate(dataloader):
        dist, labels = model.fetch_batch_distances(x)
        print(dist, labels)
    print("center:" + model.centers)


