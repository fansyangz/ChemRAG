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

current_env = check_env()
from tuning.llama3_tune_source import current_env
from embedding_datasets.milvus.insert_vector import from_word_to_embeddings, read_source
from embedding_datasets.embedding_selection.cluter_model import KMeans, Embedding_Dataset
torch.cuda.empty_cache()
path1 = f"{current_env}/embedding_datasets/data1.txt"
path2 = f"{current_env}/embedding_datasets/data2.txt"
path3 = f"{current_env}/embedding_datasets/data3.txt"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group("cluster")
    group.add_argument("--n_clusters", type=int, default=100000)
    group.add_argument("--batch_size", type=int, default=20000)
    group.add_argument("--epoch", type=int, default=100)
    group.add_argument("--file", type=int, default=1)
    args = parser.parse_args()
    n_clusters = args.n_clusters
    batch_size = args.batch_size
    epoch = args.epoch
    file = args.file
    feature = 4096
    _, total_name_count, total_name_list = read_source(path1) if file == 1 else read_source(path1, path2, path3)
    print(f"-----------------------total_name_count: {total_name_count}-----------------------")
    print("-----------------------get embedding from file to datasets-----------------------")
    datasets = Embedding_Dataset(name_list=total_name_list)
    print("-----------------------to dataloader-----------------------")
    dataloader = DataLoader(dataset=datasets, batch_size=batch_size)
    print("-----------------------model to gpu-----------------------")
    cluster_model = KMeans(n_clusters=n_clusters, n_features=feature).to("cuda")
    print("-----------------------training-----------------------")
    cluster_model(dataloader, epoch=epoch)
    print("-----------------------training success-----------------------")
    model_file_name = f"cluster_{n_clusters}-epoch_{epoch}-file_data{'1' if file == 1 else 'all'}.bin"
    torch.save(cluster_model, model_file_name)
    print("ok!!!")


