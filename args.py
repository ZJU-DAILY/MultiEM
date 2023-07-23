from dataclasses import dataclass

import tyro


@dataclass
class MainArgs:
    data_path: str = "data/processed"
    data_name: str = "Music-20"

    # selecting
    eer_flag: bool = True
    col_sim_threshold: float = 0.9  # gamma
    selection_rate: float = 0.2  # r
    # merging
    k: int = 1  # k
    min_dis: float = 0.5  # m
    # pruning
    eps: float = 1.0  # epsilon
    # parallel
    run_in_parallel: bool = False

    lm_model_or_path: str = "all-MiniLM-L12-v2"
    device: str = "cuda"
    seed: int = 3407
    max_seq_length: int = 64
    batch_size: int = 512


def build_main_args():
    return tyro.cli(MainArgs)
