import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from embedding_datasets.milvus.insert_vector import from_word_to_embeddings


class KMeans(nn.Module):
    def __init__(self, n_clusters, n_features):
        super(KMeans, self).__init__()
        self.n_clusters = n_clusters
        self.centers = nn.Parameter(torch.randn(n_clusters, n_features))

    def forward(self, dataloader, epoch=100):
        last_dist = 0
        for batch in range(epoch):
            total_dist = 0
            for en, x in enumerate(dataloader):
                print(f"batch:{batch}/{epoch}, inner:{en}/{len(dataloader)}")
                dist, labels = self.fetch_batch_distances(x)
                total_dist = total_dist + dist.sum().item()
                with torch.no_grad():
                    for i in range(self.n_clusters):
                        cluster_points = x[labels == i]
                        if len(cluster_points) > 0:
                            self.centers[i] = torch.mean(cluster_points, dim=0)
                del dist
                del labels
                del cluster_points
            torch.cuda.empty_cache()
            this_dist = total_dist/len(dataloader)
            print(f"batch: {batch} success,avg_dist: {this_dist}")
            if this_dist == last_dist:
                break
            else:
                last_dist = this_dist

    def fetch_batch_distances(self, batch):
        distances = torch.cdist(batch, self.centers)
        return torch.min(distances, dim=1)


class Embedding_Dataset(Dataset):
    def __init__(self, name_list):
        self.name_list = name_list

    def __getitem__(self, index):
        return from_word_to_embeddings(word=self.name_list[index], return_json=False)

    def __len__(self):
        return len(self.name_list)


