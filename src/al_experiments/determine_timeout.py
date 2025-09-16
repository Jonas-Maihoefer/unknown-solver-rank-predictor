import os
import time
from al_experiments.accuracy import Accuracy

useCupy = os.getenv("USECUDA", "0") == "1"

if useCupy:
    import cupy as np
else:
    import numpy as np


def static_timeout_5000(
        acc_calculator: Accuracy,
        solver_string: str,
        number_of_instances,
        prev_thresh
):
    return np.ascontiguousarray(
        np.full((number_of_instances,), 27, dtype=np.int32)
    )


def quantized_double_punish(
        acc_calculator: Accuracy,
        solver_string: str,
        number_of_instances,
        prev_thresh
):
    start = time.time_ns()
    max_acc = 0
    min_diff = 999999999.0

    ###### check if cond already met
    runtime_frac, cross_acc, stability = acc_calculator.sample_result(
        prev_thresh, acc_calculator.pred,
        "discard"
    )
    if runtime_frac > acc_calculator.break_after_runtime_fraction:
        return prev_thresh
    if acc_calculator.thresh_breaking_condition.fn(runtime_frac, cross_acc, stability):
        return prev_thresh
    #########

    while True:
        thresholds, max_acc, min_diff = acc_calculator.add_runtime_quantized(
            prev_thresh, max_acc, min_diff
        )
        if min_diff == -1:
            break
    print(f"took {(time.time_ns() - start) / 1_000_000_000}s")

    return thresholds


def quantized_mean_punish(
        acc_calculator: Accuracy,
        solver_string: str,
        number_of_instances,
        prev_thresh
):
    # precalculate pred
    mean_par_2 = acc_calculator.get_remaining_mean(prev_thresh).mean()
    print(f"mean par_2_score is {mean_par_2}")
    acc_calculator.pred = np.full((27,), mean_par_2)
    start = time.time_ns()
    max_acc = 0
    min_diff = 999999999.0

    ###### check if cond already met
    runtime_frac, cross_acc, stability = acc_calculator.sample_result(
        prev_thresh, acc_calculator.pred,
        "discard"
    )
    if runtime_frac > acc_calculator.break_after_runtime_fraction:
        return prev_thresh
    if acc_calculator.thresh_breaking_condition.fn(runtime_frac, cross_acc, stability):
        return prev_thresh
    #########

    while True:
        thresholds, max_acc, min_diff = acc_calculator.add_runtime_quantized_mean_punish(
            prev_thresh, max_acc, min_diff
        )
        if min_diff == -1:
            break
    print(f"took {(time.time_ns() - start) / 1_000_000_000}s")

    return thresholds


def create_instance_wise(delta):
    def instance_wise(
            acc_calculator: Accuracy,
            solver_string: str,
            number_of_instances,
            prev_thresh
    ):
        return acc_calculator.instance_wise_timeout(delta)

    return instance_wise
