from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from functional import pseq

from log import log


@dataclass()
class Table:
    idx: str
    tids: List[int]
    tuple_ids: List[int]

    def get_tuples(self, min_cnt=1):
        res = pseq(zip(self.tids, self.tuple_ids))\
            .group_by(lambda x: x[1])\
            .map(lambda x: x[1])\
            .map(lambda x: [xi[0] for xi in x])\
            .filter(lambda x: len(x) > min_cnt)\
            .map(lambda x: sorted(x))\
            .map(lambda x: tuple(x))\
            .to_list()
        return res


def read_table(data_path: Path, selected_attrs=None):
    if selected_attrs is None:
        table = pd.read_csv(
            data_path, dtype={"postcode": str})
    else:
        table = pd.read_csv(
            data_path, dtype={"postcode": str}, usecols=selected_attrs)
    return table


def read_all_tables(data_path: Path, num=-1, selected_attrs=None) -> Tuple[int, List[pd.DataFrame]]:
    log(f"selected_attrs: {selected_attrs}")
    i = 0
    tables = []
    while (data_path / f"table_{i}.csv").is_file():
        table = read_table(data_path / f"table_{i}.csv", selected_attrs)
        tables.append(table)
        i += 1
        if i == num:
            break
    return i, tables


def read_ground_truth(data_path: Path) -> List[Tuple[int]]:
    with (data_path / "ground_truth.txt").open("r") as rd:
        return [tuple(map(int, line.split(","))) for line in rd]


def read_pair_ground_truth(data_path: Path, i: int, j: int) -> List[Tuple[int]]:
    with (data_path / f"ground_truth_{i}_{j}.txt").open("r") as rd:
        return [tuple(map(int, line.split(","))) for line in rd]


def textify_table(table: pd.DataFrame):
    sentences = table.iloc[:, 1:] \
        .astype(str) \
        .apply(lambda x: x + " ") \
        .values.sum(axis=1)\
        .tolist()
    return sentences
