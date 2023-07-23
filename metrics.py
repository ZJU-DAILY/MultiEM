from dataclasses import dataclass
from itertools import combinations, chain
from typing import Optional, List, Tuple

from log import log


@dataclass()
class Metric:
    p: Optional[float] = None
    r: Optional[float] = None
    f1: Optional[float] = None

    def log(self, prefix=None):
        log(f"{f'[{prefix}] ' if prefix else ''}P={self.p:.4f}, R={self.r:.4f}, F1={self.f1:.4f}")


def evaluate_f1(ground_truth: List[Tuple], prediction: List[Tuple]) -> Metric:
    ground_truth = set(ground_truth)
    prediction = set(prediction)
    truth = len(ground_truth.intersection(prediction))
    P = truth / len(prediction)
    R = truth / len(ground_truth)
    F1 = 2 * P * R / (P + R)
    return Metric(p=P, r=R, f1=F1)


def tuple_2_pairs(tuples: List[Tuple]) -> List[Tuple]:
    return list(chain(*[combinations(tup, 2) for tup in tuples]))


def evaluate_pair_f1(ground_truth: List[Tuple], prediction: List[Tuple]) -> Metric:
    ground_truth_pairs = tuple_2_pairs(ground_truth)
    prediction_pairs = tuple_2_pairs(prediction)
    return evaluate_f1(ground_truth_pairs, prediction_pairs)


def evaluate(ground_truth: List[Tuple], prediction: List[Tuple]) -> Tuple[Metric, Metric]:
    log(f"num of ground truth: {len(ground_truth)}")
    log(f"num of prediction: {len(prediction)}")
    f1_metric = evaluate_f1(ground_truth, prediction)
    pair_f1_metric = evaluate_pair_f1(ground_truth, prediction)
    return f1_metric, pair_f1_metric


def evaluate_log(ground_truth: List[Tuple], prediction: List[Tuple]):
    metric, pair_metric = evaluate(ground_truth, prediction)
    metric.log(prefix="none")
    pair_metric.log(prefix="pair")
