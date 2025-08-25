import os
import pandas as pd
import re
import random
import pickle
import subprocess
from al_experiments.determine_timeout import quantized_mean_punish, quantized_double_punish, static_timeout_5000
from al_experiments.experiment_config import ExperimentConfig
from al_experiments.accuracy import Accuracy, create_cross_acc_breaking, create_softmax_fn, greedy_cross_acc, greedy_rmse, knapsack_rmse, select_best_idx
from scipy.interpolate import interp1d

from al_experiments.plot_generator import PlotGenerator
from al_experiments.instance_selector import InstanceSelector, choose_instances_random, variance_based_selection_1, variance_based_selection_2, highest_rt_selection, lowest_variance, lowest_variances_per_rt, lowest_rt_selection
from al_experiments.constants import Constants

useCupy = os.getenv("USECUDA", "0") == "1"

print(f"use cuda: {os.getenv('USECUDA', 'not set')}")

if useCupy:
    import cupy as np
else:
    import numpy as np

# global config
break_after_solvers = 200
break_after_runtime_fraction = 2  # 0.655504  # determined by 0e993e00
total_samples = 500  # max is 5354 because of sample_result_after_instances
# total_runtime = 25860323 s
# global results
all_results = []
all_timeout_results = []
all_instance_selection_results = {}
plot_generator = None

# experiment config
experiment_configs = ExperimentConfig(
    determine_thresholds=quantized_double_punish,
    select_idx=select_best_idx,
    scoring_fn=greedy_cross_acc,  # knapsack_rmse, greedy_rmse, knapsack_cross_acc, greedy_cross_acc
    thresh_breaking_condition=create_cross_acc_breaking(1.1),
    temperatures=[],  # [0.5, 0.35, 0.25, 0.125, 0.09, 0.06125, 0.03075, 0.01530, 0.008, 0.004],
    rt_weights=[1],   # [1.0, 0.95, 1.1, 1.3, 1.5, 0.8, 1.6, 1.2, 1.4, 1.7, 1.05, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0],
    instance_selections=[choose_instances_random, variance_based_selection_1, variance_based_selection_2, highest_rt_selection, lowest_variance, lowest_variances_per_rt, lowest_rt_selection],
    individual_solver_plots=True
)


def get_git_commit_hash():
    # runs: git rev-parse HEAD
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=True,  # raises CalledProcessError on non-zero exit
    )
    # decode bytes to str, strip newline
    return result.stdout.decode("utf-8").strip()[:8]


def convert_to_sorted_runtimes(runtimes: pd.DataFrame):
    runtimes = runtimes.values

    sorted_idx_list = []
    sorted_runtimes_list = []

    for row in runtimes:
        inner_idx_list = []
        inner_sorted_rts = []

        # get sorted indices based on runtime
        sorted_idx = np.argsort(row)

        # print(sorted_idx)

        # add zero element
        inner_idx_list.insert(0, -1)
        inner_sorted_rts.insert(0, 0.0)

        for i in sorted_idx:
            inner_idx_list.append(int(i))
            inner_sorted_rts.append(float(row[i]))

        sorted_idx_list.append(inner_idx_list)
        sorted_runtimes_list.append(inner_sorted_rts)

    sorted_idx = np.array(sorted_idx_list, dtype=int)

    #print(sorted_idx)
    sorted_rt = np.array(sorted_runtimes_list)

    return sorted_idx, sorted_rt


def determine_runtime_fraction(df: pd.DataFrame, runtime_limits: pd.Series):
    # set infinite value to timeout
    df_non_inf = df.replace([np.inf, -np.inf], 5000)

    total_runtime = 0
    used_runtime = 0

    for index, row in df_non_inf.iterrows():
        total_runtime += row.sum()

        row[row > runtime_limits[index]] = runtime_limits[index]

        used_runtime += row.sum()

    runtime_fraction = used_runtime/total_runtime
    print(f"the runtime fraction is {runtime_fraction}")


def vec_to_runtime_frac(
        thresholds,
        runtimes,
        total_runtime: float
):
    """
    thresholds: 1D array-like of shape (5355,)
    runtimes:    2D array-like of shape (5355, 28)

    For each i, any runtimes[i, j] > thresholds[i]
    is replaced bythresholds[i],
    then everything is summed and compared to the given total_runtime
    """
    thresholds = np.ascontiguousarray(thresholds, dtype=np.float32)
    runtimes = np.ascontiguousarray(runtimes,  dtype=np.float32)

    used_runtime = np.minimum(runtimes, thresholds[:, None]).sum()

    return used_runtime / total_runtime


def vec_to_single_runtime_frac(
        thresholds,
        runtimes,
        index: int
):
    """
    thresholds: 1D array-like of shape (5355,)
    runtimes:    2D array-like of shape (5355, 28)

    For each i, any runtimes[i, j] > thresholds[i]
    is replaced by thresholds[i],
    then everything is summed and compared to the given total_runtime
    """
    thresholds = np.ascontiguousarray(thresholds, dtype=np.float32)
    runtimes = np.ascontiguousarray(runtimes,  dtype=np.float32)

    solver_runtimes = runtimes[:, index]

    total_runtime = solver_runtimes.sum()

    used_runtime = np.minimum(solver_runtimes, thresholds).sum()

    return used_runtime / total_runtime


def store_and_get_mean_result(rt_weight, temp):

    weight_string = re.sub(r'[^0-9a-zA-Z_]', '_', str(rt_weight))
    temp_string = re.sub(r'[^0-9a-zA-Z_]', '_', str(temp))

    rs_string = ""

    # construct df
    df = pd.DataFrame.from_records(all_results)
    df.to_pickle(f"./pickle/{git_hash}_rt_weigth_{weight_string}_temp_{temp_string}.pkl.gz", compression="gzip")

    plot_generator.plot_avg_results(df, total_samples)

    return rs_string


def get_stats(df_rated, df_runtimes, par_2_scores_series, par_2_scores, runtimes, thresholds, solver_index, acc_calculator: Accuracy, progress: str):
    runtimes_rated = np.ascontiguousarray(
        df_rated.copy(), dtype=np.float32
    )
    diff = acc_calculator.vec_to_diff(thresholds, runtimes_rated, par_2_scores, par_2_scores.mean(), solver_index)
    par_2_scores = np.ascontiguousarray(
        par_2_scores_series, dtype=np.float32
    )
    runtimes_unrated = np.ascontiguousarray(
        df_runtimes.copy(), dtype=np.float32
    )
    true_acc = acc_calculator.vec_to_true_acc(
        thresholds, runtimes_rated, par_2_scores, solver_index
    )
    cross_acc = acc_calculator.vec_to_cross_acc(thresholds, runtimes_rated, par_2_scores)
    runtime_frac = vec_to_single_runtime_frac(thresholds, runtimes_unrated, solver_index)
    print(f"{progress} cross accuracy is {cross_acc}")
    print(f"{progress} true acc would be {true_acc}")
    print(f"{progress} difference of both scores is {diff}")

    return {"runtime_frac": runtime_frac, "cross_acc": cross_acc, "true_acc": true_acc, "diff": diff}


def run_multi_temp_experiments(experiment_config: ExperimentConfig):
    if experiment_config.select_idx.__name__ != 'create_softmax_fn':
        run_multi_rt_weights_experiments(experiment_config)
        return

    tot_res_str = ''
    for temp in experiment_config.temperatures:
        experiment_config.select_idx = create_softmax_fn(temp)
        tot_res_str += run_multi_rt_weights_experiments(experiment_config, temp)

    print("total result string:")
    print(tot_res_str)


def run_multi_rt_weights_experiments(experiment_config: ExperimentConfig, temp=None):
    global plot_generator
    results_string = ""

    for rt_weight in experiment_config.rt_weights:
        if len(experiment_config.rt_weights) == 1 and len(experiment_config.temperatures) == 1:
            plot_generator = PlotGenerator(git_hash, experiment_config)
        else:
            if temp is None:
                plot_generator = PlotGenerator(git_hash, experiment_config, f"rt_weight_{rt_weight}")
            else:
                plot_generator = PlotGenerator(git_hash, experiment_config, f"rt_weight_{rt_weight}_temp_{temp}")
                print(f"running with a temperature of {temp}")
        print(f"running with a runtime weight of {rt_weight}")
        # reset results
        global all_results
        all_results = []
        results_string += run_experiment(experiment_config, rt_weight, temp)
        print("results so far:")
        print(results_string)

    return results_string


def run_experiment(experiment_config: ExperimentConfig, rt_weight, temp):
    with open(
        "../al-for-sat-solver-benchmarking-data/pickled-data/anni_full_df.pkl",
        "rb"
    ) as file:
        df: pd.DataFrame = pickle.load(file).copy()

    print(df)

    # min value of df > 0.0 is 0.003021
    # clean the 3 runtimes of 0.0s
    df = df.replace(0.0, 0.0001)

    solver_names = df.columns.tolist()

    # plot_generator.visualize_predictions_exclude_instances(df_rated, df_runtimes)

    random_solver_order = list(range(28))

    random.shuffle(random_solver_order)

    for index, solver_index in enumerate(random_solver_order):
        solver_string = f"{solver_index}_" + \
                        f"{solver_names[solver_index]}"

        if index >= break_after_solvers:
            break

        print(f"removing solver {solver_names[solver_index]}")

        # reduced values
        df_reduced = df.drop(solver_names[solver_index], axis=1)
        total_runtime = df_reduced.replace([np.inf, -np.inf], 5000).stack().sum()
        total_rt_removed_solver = df.replace([np.inf, -np.inf], 5000)[solver_names[solver_index]].sum()
        mean_rt = df_reduced.replace([np.inf, -np.inf], 5000).mean(axis=1)
        df_reduced_cleaned = df_reduced.loc[mean_rt != 5000.0].copy()
        reduced_df_runtimes = df_reduced_cleaned.replace([np.inf, -np.inf], 5000)
        reduced_df_rated = df_reduced_cleaned.replace([np.inf, -np.inf], 10000)
        reduced_par_2_scores_series = reduced_df_rated.mean(axis=0)
        par_2_scores = np.ascontiguousarray(
            np.asarray(reduced_par_2_scores_series.values), dtype=np.float32
        )
        mean_par_2_score = par_2_scores.mean()
        sorted_runtimes = convert_to_sorted_runtimes(reduced_df_runtimes)
        sorted_runtimes_rated = convert_to_sorted_runtimes(reduced_df_rated)

        # no reduced; USE WITH CARE!
        df_cleaned = df.loc[mean_rt != 5000.0].copy()
        df_cleaned_rated = df_cleaned.replace([np.inf, -np.inf], 10000)
        df_cleaned_runtimes = df_cleaned.replace([np.inf, -np.inf], 5000)
        cleaned_par_2_values = df_cleaned_rated.mean(axis=0)
        par_2_score_removed_solver = cleaned_par_2_values[solver_index]
        runtime_of_removed_solver = np.ascontiguousarray(
            np.asarray(df_cleaned_runtimes.iloc[:, solver_index].values),
            dtype=np.float32
        )

        # constants
        con = Constants(df_cleaned)
        sample_result_after_iterations = int(
            con.number_of_instances * (
                con.number_of_solvers - 1) / total_samples
        )
        sample_result_after_instances = int(
            con.number_of_instances / total_samples
        )

        acc_calculator = Accuracy(
            con, total_runtime,
            total_rt_removed_solver, break_after_runtime_fraction,
            sample_result_after_iterations, sorted_runtimes,
            sorted_runtimes_rated, par_2_scores, mean_par_2_score,
            par_2_score_removed_solver, runtime_of_removed_solver,
            experiment_config.select_idx,
            solver_string,
            all_results,
            experiment_config.scoring_fn,

            experiment_config.thresh_breaking_condition,
            rt_weight,
            with_remaining_mean=experiment_config.determine_thresholds.__name__ == "quantized_mean_punish",
        )

        # determine thresholds for perfect differentiation of remaining solvers
        thresholds = experiment_config.determine_thresholds(
            acc_calculator, solver_string, con.number_of_instances
        )

        print("here is the calculated threshold vector:")

        for threshold in thresholds:
            print(f"{threshold}", end=", ")
        print()

        print(f"adding solver {solver_names[solver_index]} back in")

        for instance_selection in experiment_config.instance_selections:
            print(f"select instances based on method {instance_selection.__name__}")
            if instance_selection.__name__ == "no_selection":
                continue

            selector = InstanceSelector(
                con, thresholds, sorted_runtimes, acc_calculator,
                sample_result_after_instances, instance_selection
            )

            selector.make_selection()

        if experiment_config.individual_solver_plots:
            plot_generator.plot_solver_results(
                all_results,
                solver_string
            )

    return store_and_get_mean_result(rt_weight, temp)


if __name__ == "__main__":

    git_hash = get_git_commit_hash()

    plot_generator = PlotGenerator(git_hash, experiment_configs)
    #plot_generator.create_progress_plot()

    print(f"start experiment on {git_hash}")
    run_multi_temp_experiments(experiment_configs)

    print("ended experiment")
