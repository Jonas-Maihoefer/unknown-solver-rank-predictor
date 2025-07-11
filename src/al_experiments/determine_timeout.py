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
        number_of_instances
):
    return np.ascontiguousarray(
        np.full((number_of_instances,), 27, dtype=np.int32)
    )


def quantized_min_diff(
        acc_calculator: Accuracy,
        solver_string: str,
        number_of_instances
):

    # initialize tresholds with 0
    thresholds = np.ascontiguousarray(
        np.full((number_of_instances,), 0, dtype=np.int32)
    )
    start = time.time_ns()
    max_acc = 0
    min_diff = 999999999.0

    while True:
        thresholds, max_acc, min_diff = acc_calculator.add_runtime_quantized(
            thresholds, max_acc, min_diff
        )
        if min_diff == -1:
            break
    print(f"took {(time.time_ns() - start) / 1_000_000_000}s")

    return thresholds


def quantized_mean_punish(
        acc_calculator: Accuracy,
        solver_string: str,
        number_of_instances
):

    # initialize tresholds with 0
    thresholds = np.ascontiguousarray(
        np.full((number_of_instances,), 0, dtype=np.int32)
    )
    # precalculate pred
    mean_par_2 = acc_calculator.get_remaining_mean(thresholds).mean()
    print(f"mean par_2_score is {mean_par_2}")
    acc_calculator.pred = np.full((27,), mean_par_2)
    start = time.time_ns()
    max_acc = 0
    min_diff = 999999999.0

    while True:
        thresholds, max_acc, min_diff = acc_calculator.add_runtime_quantized_mean_punish(
            thresholds, max_acc, min_diff
        )
        if min_diff == -1:
            break
    print(f"took {(time.time_ns() - start) / 1_000_000_000}s")

    return thresholds
