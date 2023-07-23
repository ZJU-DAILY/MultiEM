from typing import List, Tuple

import numpy as np
from joblib import Parallel, delayed
from sklearn.cluster._dbscan_inner import dbscan_inner
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from args import MainArgs


class Pruner:
    def __init__(self, eps=0.5, minEnt=2):
        self.eps = eps
        self.minEnt = minEnt

    def fit_predict(self, X):
        nn_model = NearestNeighbors(radius=self.eps, metric="euclidean")
        nn_model.fit(X)
        nbs = nn_model.radius_neighbors(X, return_distance=False)
        n_neighbors = np.array([len(neighbors) for neighbors in nbs])
        labels = np.full(X.shape[0], -1, dtype=np.intp)
        core_samples = np.asarray(n_neighbors >= self.minEnt, dtype=np.uint8)
        dbscan_inner(core_samples, nbs, labels)
        return labels


def pruning_item(all_embeddings, item, pruner: Pruner):
    labels = pruner.fit_predict(all_embeddings[np.array(list(item))])
    new_item = tuple([x for x, l in zip(item, labels) if l != -1])
    return new_item if len(new_item) > 1 else None


def pruning(prediction: List[Tuple], all_embeddings: np.array, args: MainArgs) -> List[Tuple]:
    pruner = Pruner(eps=args.eps, minEnt=2)
    new_prediction = [pruning_item(all_embeddings, item, pruner)
                      for item in tqdm(prediction)]
    new_prediction = [x for x in new_prediction if x]
    return new_prediction


def pruning_parallel(prediction: List[Tuple], all_embeddings: np.array, args: MainArgs) -> List[Tuple]:
    pruner = Pruner(eps=args.eps, minEnt=2)
    new_prediction = Parallel(n_jobs=-1)(
        delayed(pruning_item)(all_embeddings, item, pruner) for item in tqdm(prediction))
    new_prediction = [x for x in new_prediction if x]
    return new_prediction
