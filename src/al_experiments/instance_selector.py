import os

import pandas as pd
from al_experiments.accuracy import Accuracy
from al_experiments.constants import number_of_reduced_solvers, number_of_instances, instance_idx, idx, rt


useCupy = os.getenv("USECUDA", "0") == "1"

if useCupy:
    import cupy as np
else:
    import numpy as np


class InstanceSelector:

    def __init__(
            self,
            thresholds,
            sorted_rt,
            acc_calculator: Accuracy,
            sample_intervall: int,
            choosing_fn
    ):
        self.thresholds = thresholds
        self.sorted_rt = sorted_rt
        self.acc_calculator = acc_calculator
        self.sample_intervall = sample_intervall
        self.choosing_fn = choosing_fn
        self.n = 0
        self.total_runtime = 0
        self.choosen_instances = []
        self.results = []
        self.choosen_thresholds = np.ascontiguousarray(
            np.full((number_of_instances,), 0), dtype=np.int32
        )
        self.pred = np.ascontiguousarray(
            np.full((number_of_reduced_solvers,), 0), dtype=np.float32
        )

    def make_selection(self):
        not_choosen_yet = ~np.isin(instance_idx, self.choosen_instances)
        has_timeout = self.thresholds != 0
        combined_mask = not_choosen_yet & has_timeout
        possible_instances = instance_idx[combined_mask]

        while (len(possible_instances) > 0):
            #print("possible instances")
            #print(possible_instances)
            new_instance = self.choosing_fn(
                possible_instances, self.thresholds, self.sorted_rt
            )
            included_solvers = self.thresholds[new_instance]
            self.choosen_thresholds[new_instance] = included_solvers
            runtimes = self.sorted_rt[rt][new_instance]
            idxs = self.sorted_rt[idx][new_instance]
            timeout = runtimes[included_solvers]
            #print(f"last included solver is solver {solver}")
            included_idxs = idxs[:included_solvers + 1]
            included_runtimes = runtimes[:included_solvers + 1]
            #print("included")
            #print(included)
            excluded_idxs = idxs[included_solvers + 1:]
            #print("excluded")
            #print(excluded)

            if timeout == 5000:
                included_runtimes[included_runtimes == 5000] = 10000

            self.pred[included_idxs] += included_runtimes
            self.pred[excluded_idxs] += timeout * 2

            if self.n % self.sample_intervall == 0:
                self.results.append(
                    self.acc_calculator.sample_result(
                        self.choosen_thresholds, self.pred
                    )
                )
            self.n += 1

            # prepare next loop
            self.choosen_instances.append(new_instance)
            not_choosen_yet = ~np.isin(instance_idx, self.choosen_instances)
            has_timeout = self.thresholds != 0
            combined_mask = not_choosen_yet & has_timeout
            possible_instances = instance_idx[combined_mask]

        # sample last result
        self.results.append(
            self.acc_calculator.sample_result(
                self.choosen_thresholds, self.pred
            )
        )


def choose_instances_random(
        possible_instances,
        thresholds,
        sorted_runtimes
):
    return np.random.choice(possible_instances)


def variance_based_selection_1(
        possible_instances,
        thresholds,
        sorted_runtimes
):
    """this methods works slightly better (tested in e99fb452 (version 2) vs 697d3971 (this version))"""
    timeouts = sorted_runtimes[rt][instance_idx, thresholds[instance_idx]]
    runtimes = sorted_runtimes[rt].copy()

    runtimes[runtimes > timeouts[:, None]] = np.nan
    runtimes[:, 0] = np.nan
    runtimes[runtimes == 0.0] = 0.001

    variances = np.nanvar(runtimes, axis=1)
    means = np.nanmean(runtimes, axis=1)

    score = variances/means

    score = score[possible_instances]

    best_idx = possible_instances[np.nanargmax(score)]

    return best_idx


def variance_based_selection_2(
        possible_instances,
        thresholds,
        sorted_runtimes
):
    """this methods works slightly worse than `variance_based_selection_1` (tested in e99fb452 (this version) vs 697d3971 (version 1))"""
    timeouts = sorted_runtimes[rt][instance_idx, thresholds[instance_idx]]

    scores = sorted_runtimes[rt].copy()
    runtimes = sorted_runtimes[rt].copy()

    scores[:, 0] = np.nan
    scores[scores == 0.0] = 0.001
    scores[scores == 5000.0] = 10000.0
    scores = np.where(
        scores >= timeouts[:, None],
        timeouts[:, None] * 2,
        scores
    )

    runtimes[:, 0] = np.nan
    runtimes[runtimes == 0.0] = 0.001

    runtimes = np.where(
        runtimes >= timeouts[:, None],
        timeouts[:, None],
        runtimes
    )

    variances = np.nanvar(scores, axis=1)
    mean_rts = np.nanmean(runtimes, axis=1)

    score = variances/mean_rts

    score = score[possible_instances]

    if np.isnan(score).all():
        best_idx = 0
    else:
        best_idx = np.nanargmax(score)

    best_idx = possible_instances[best_idx]

    return best_idx
