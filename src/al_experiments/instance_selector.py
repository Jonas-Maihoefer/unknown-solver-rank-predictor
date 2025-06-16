from statistics import pstdev
import numpy as np

from al_experiments.accuracy import Accuracy

number_of_instances = 5355
instance_idx = np.arange(number_of_instances)


class InstanceSelector:
    number_of_instances = 5355
    instance_idx = np.arange(number_of_instances)
    number_of_solvers = 28
    number_of_reduced_solvers = 27

    def __init__(
            self,
            thresholds: np.ndarray,
            sorted_rt: np.ndarray,
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
            np.full((self.number_of_instances,), 0), dtype=np.int32
        )
        self.pred = np.ascontiguousarray(
            np.full((self.number_of_reduced_solvers,), 0), dtype=np.float32
        )

    def make_selection(self):
        not_choosen_yet = ~np.isin(self.instance_idx, self.choosen_instances)
        has_timeout = self.thresholds != 0
        combined_mask = not_choosen_yet & has_timeout
        possible_instances = self.instance_idx[combined_mask]

        while (len(possible_instances) > 0):
            #print("possible instances")
            #print(possible_instances)
            new_instance = self.choosing_fn(
                possible_instances, self.thresholds, self.sorted_rt
            )
            included_solvers = self.thresholds[new_instance]
            self.choosen_thresholds[new_instance] = included_solvers
            runtimes = self.sorted_rt[new_instance]
            solver, timeout = runtimes[included_solvers]
            #print(f"last included solver is solver {solver}")
            included = runtimes[:included_solvers + 1]
            #print("included")
            #print(included)
            excluded = runtimes[included_solvers + 1:]
            #print("excluded")
            #print(excluded)

            if timeout == 5000:
                included['runtime'][included['runtime'] == 5000] = 10000

            self.pred[included['idx']] += included['runtime']
            self.pred[excluded['idx']] += timeout * 2

            if self.n % self.sample_intervall == 0:
                self.results.append(
                    self.acc_calculator.sample_result(
                        self.choosen_thresholds, self.pred
                    )
                )
            self.n += 1

            # prepare next loop
            self.choosen_instances.append(new_instance)
            not_choosen_yet = ~np.isin(self.instance_idx, self.choosen_instances)
            has_timeout = self.thresholds != 0
            combined_mask = not_choosen_yet & has_timeout
            possible_instances = self.instance_idx[combined_mask]

        # sample last result
        self.results.append(
            self.acc_calculator.sample_result(
                self.choosen_thresholds, self.pred
            )
        )


def choose_instances_random(
        possible_instances: np.ndarray,
        thresholds: np.ndarray,
        sorted_runtimes: np.ndarray
):
    return np.random.choice(possible_instances)


def variance_based_selection_1(
        possible_instances: np.ndarray,
        thresholds: np.ndarray,
        sorted_runtimes: np.ndarray
):
    """this methods works slightly better (tested in e99fb452 (version 2) vs 697d3971 (this version))"""
    timeouts = sorted_runtimes['runtime'][instance_idx, thresholds[instance_idx]]
    runtimes = sorted_runtimes['runtime'].copy()

    runtimes[runtimes > timeouts[:, None]] = np.nan
    runtimes[:, 0] = np.nan
    runtimes[runtimes == 0.0] = 0.001

    variances = np.nanvar(runtimes, axis=1)
    means = np.nanmean(runtimes, axis=1)

    score: np.ndarray = variances/means

    score = score[possible_instances]

    best_idx = possible_instances[np.nanargmax(score)]

    return best_idx


def variance_based_selection_2(
        possible_instances: np.ndarray,
        thresholds: np.ndarray,
        sorted_runtimes: np.ndarray
):
    """this methods works slightly worse than `variance_based_selection_1` (tested in e99fb452 (this version) vs 697d3971 (version 1))"""
    timeouts = sorted_runtimes['runtime'][instance_idx, thresholds[instance_idx]]
    scores = sorted_runtimes['runtime'].copy()
    runtimes = sorted_runtimes['runtime'].copy()

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

    score: np.ndarray = variances/mean_rts

    score = score[possible_instances]

    best_idx = possible_instances[np.nanargmax(score)]

    return best_idx
