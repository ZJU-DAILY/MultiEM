from typing import List
from copy import deepcopy

import numpy as np
import pandas as pd
import networkx as nx
from joblib import Parallel, delayed

from data import Table
from args import MainArgs
from log import log
from timer import Timer
from utils import knn_search, shuffle


def get_table_embeddings(table: Table, all_embeddings: np.array):
    embeddings = all_embeddings[table.tids]
    df = pd.DataFrame(embeddings)
    df["group"] = table.tuple_ids
    mean_embeddings = df.groupby('group').mean().to_numpy()
    return mean_embeddings


def search_ij(embeddings_i: np.array, embeddings_j: np.array, k: int, seed: int, min_dis: float):
    ids_i = list(range(embeddings_i.shape[0]))
    ids_j = list(range(embeddings_j.shape[0]))
    I1, D1 = knn_search(embeddings_j, np.array(
        ids_j), embeddings_i, k, seed)
    pairs_ij = [(p, vi) for p, v, d in zip(ids_i, I1, D1)
                for vi, di in zip(v, d) if di <= min_dis]
    return pairs_ij


def merge_ij(table_i: Table, table_j: Table, all_embeddings: np.array, args: MainArgs) -> Table:
    timer = Timer()
    idx_i, idx_j = table_i.idx, table_j.idx
    log(f"table {idx_i}, {idx_j}")
    timer.start()
    embeddings_i = get_table_embeddings(table_i, all_embeddings)
    embeddings_j = get_table_embeddings(table_j, all_embeddings)
    tm = timer.stop()
    log(f"get embeddings: {tm}")
    timer.start()
    pairs_ij = search_ij(embeddings_i, embeddings_j,
                         args.k, args.seed, args.min_dis)
    pairs_ji = search_ij(embeddings_j, embeddings_i,
                         args.k, args.seed, args.min_dis)
    tm = timer.stop()
    log(f"ann search: {tm}")
    timer.start()
    pairs_ji = [(x[1], x[0]) for x in pairs_ji]
    pairs = set(pairs_ij).intersection(set(pairs_ji))
    size_i = int(embeddings_i.shape[0])
    size_j = int(embeddings_j.shape[0])
    edges = [(x[0], int(x[1]+size_i)) for x in pairs]
    g = nx.Graph()
    g.add_edges_from(edges)
    new_tids = []
    new_tuple_ids = []
    new_tuple_cnt = 0
    app_tids = set()
    df_i = pd.DataFrame(table_i.tids)
    df_i["group"] = table_i.tuple_ids
    gi = df_i.groupby('group')[0].apply(list).to_dict()
    df_j = pd.DataFrame(table_j.tids)
    df_j["group"] = table_j.tuple_ids
    gj = df_j.groupby('group')[0].apply(list).to_dict()
    for c in nx.connected_components(g):
        new_tuple = []
        for ci in c:
            if ci < size_i:
                new_tuple.extend(gi[ci])
            else:
                new_tuple.extend(gj[ci-size_i])
        assert len(new_tuple) > 0
        app_tids.update(c)
        new_tids.extend(new_tuple)
        new_tuple_ids.extend([new_tuple_cnt]*len(new_tuple))
        new_tuple_cnt += 1
    rem_tuple_ids = [x for x in range(
        size_i+size_j) if x not in app_tids]
    for rem_tuple_id in rem_tuple_ids:
        if rem_tuple_id < size_i:
            new_tuple = gi[rem_tuple_id]
        else:
            new_tuple = gj[rem_tuple_id-size_i]
        assert len(new_tuple) > 0
        new_tids.extend(new_tuple)
        new_tuple_ids.extend([new_tuple_cnt]*len(new_tuple))
        new_tuple_cnt += 1
    tm = timer.stop()
    log(f"new table: {tm}")
    new_table = Table(f"{idx_i}-{idx_j}", new_tids, new_tuple_ids)
    return new_table


def merge(tables: List[Table], all_embeddings: np.array, args: MainArgs) -> Table:
    cur_tables = [deepcopy(table) for table in tables]
    while len(cur_tables) > 1:
        new_tables = []
        n = len(cur_tables)
        cur_tables = shuffle(cur_tables, args.seed)
        index_i = 0
        while index_i + 1 < n:
            table_i = cur_tables[index_i]
            table_j = cur_tables[index_i + 1]
            new_table = merge_ij(table_i, table_j, all_embeddings, args)
            new_tables.append(new_table)
            index_i += 2
        if index_i == n - 1:
            new_tables.append(cur_tables[index_i])
        cur_tables = new_tables
    assert len(cur_tables) == 1
    return cur_tables[0]


def merge_parallel(tables: List[Table], all_embeddings: np.array, args: MainArgs) -> Table:
    cur_tables = [deepcopy(table) for table in tables]
    while len(cur_tables) > 1:
        n = len(cur_tables)
        cur_tables = shuffle(cur_tables, args.seed)
        new_tables = []
        table_id_pairs = [(i, i+1) for i in range(0, n, 2) if i+1 < n]

        def fun(id_i, id_j):
            table_i = cur_tables[id_i]
            table_j = cur_tables[id_j]
            new_table = merge_ij(table_i, table_j, all_embeddings, args)
            return new_table
        new_tables = Parallel(n_jobs=len(table_id_pairs))(
            delayed(fun)(p[0], p[1]) for p in table_id_pairs)
        if n % 2 == 1:
            new_tables.append(cur_tables[-1])
        cur_tables = new_tables
    assert len(cur_tables) == 1
    return cur_tables[0]
