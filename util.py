import numpy as np
import pickle

def batch_func(iterable, batch_size):
    length = len(iterable)
    for i in range(0, length, batch_size):
        yield iterable[i:min(i + batch_size, length)]


def list_unique(str_list):
    return np.unique(np.array([s for s in str_list if s != ''])).tolist()

def load_by_name(name):
    with open(name, "rb") as file:
        obj = pickle.load(file)
    return obj