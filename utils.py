import random

import hnswlib
import numpy as np


def knn_search(value: np.array, ids: np.array, query: np.array, k: int, seed: int, metric="cosine", dim=384):
    index = hnswlib.Index(space=metric, dim=dim)
    index.init_index(max_elements=len(value),
                     ef_construction=200, M=32, random_seed=seed)
    index.add_items(value, ids)
    index.set_ef(200)
    I, D = index.knn_query(query, k=k)
    return I, D


def shuffle(x, seed):
    random.Random(seed).shuffle(x)
    return x


def element_wise_cosine_sim(a: np.array, b: np.array):
    return np.sum(a*b, axis=-1)/(np.linalg.norm(a, axis=-1)*np.linalg.norm(b, axis=-1))
