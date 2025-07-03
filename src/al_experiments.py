import os
import pandas as pd
import re
import random
import pickle
import subprocess
from al_experiments.determine_timeout import quantized_min_diff, static_timeout_5000
from al_experiments.experiment_config import ExperimentConfig
from al_experiments.accuracy import Accuracy, create_softmax_fn, select_best_idx
from scipy.interpolate import interp1d

from al_experiments.plot_generator import PlotGenerator
from al_experiments.instance_selector import InstanceSelector, choose_instances_random, variance_based_selection_1, variance_based_selection_2, lowest_rt_selection
from al_experiments.constants import number_of_solvers, number_of_instances

useCupy = os.getenv("USECUDA", "0") == "1"

print(f"use cuda: {os.getenv('USECUDA', 'not set')}")

if useCupy:
    import cupy as np
else:
    import numpy as np

# global config
break_after_solvers = 200
break_after_runtime_fraction = 0.4  # 0.655504  # determined by 0e993e00
total_samples = 500  # max is 5354 because of sample_result_after_instances
sample_result_after_iterations = int(number_of_instances * (number_of_solvers - 1) / total_samples)
sample_result_after_instances = int(number_of_instances / total_samples)
# total_runtime = 25860323 s
# global results
all_timeout_results = []
all_instance_selection_results = {}
plot_generator = None

# experiment config
experiment_configs = ExperimentConfig(
    determine_thresholds=quantized_min_diff,
    select_idx=select_best_idx,
    temperatures=[],  # [0.5, 0.35, 0.25, 0.125, 0.09, 0.06125, 0.03075, 0.01530, 0.008, 0.004],
    rt_weights=[1],
    instance_selections=[choose_instances_random, variance_based_selection_1],
    individual_solver_plots=False
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


def compute_average_grid(list_of_dfs, grid_size=total_samples):
    """
    Given a list of DataFrames, each with 'runtime_frac' and measurements
    ['cross_acc', 'true_acc', 'diff'], interpolate each onto a common grid
    and compute the average of each measurement at each grid point.
    """
    # Determine global min and max of runtime_frac across all DataFrames
    max_min_frac = max(df['runtime_frac'].min() for df in list_of_dfs)
    min_max_frac = min(df['runtime_frac'].max() for df in list_of_dfs)

    # Create an evenly spaced grid
    grid = np.linspace(max_min_frac, min_max_frac, grid_size)

    measurements = ['cross_acc', 'true_acc', 'diff']

    # Collect interpolated values for each measurement from each DataFrame
    interp_values = {m: [] for m in measurements}
    for df in list_of_dfs:
        # Create interpolation functions for each measurement
        fns = {
            m: interp1d(
                df['runtime_frac'],
                df[m],
                kind='linear',
                bounds_error=False,
                fill_value=np.nan
            ) for m in measurements
        }
        # Evaluate and store interpolated values on the grid
        for m in measurements:
            interp_values[m].append(fns[m](grid))

    # Build the averaged DataFrame
    avg_data = {'runtime_frac': grid}
    for m in measurements:
        # Stack all arrays and compute mean, ignoring NaNs
        stacked = np.vstack(interp_values[m])
        avg_data[m] = np.nanmean(stacked, axis=0)

    avg_df = pd.DataFrame(avg_data)
    return avg_df


def store_and_get_mean_result(rt_weight, temp):

    weight_string = re.sub(r'[^0-9a-zA-Z_]', '_', str(rt_weight))
    temp_string = re.sub(r'[^0-9a-zA-Z_]', '_', str(temp))

    rs_string = ""

    if not all_timeout_results[0].empty:
        avg_timeout_results = compute_average_grid(all_timeout_results, grid_size=total_samples)
        for param in ["runtime_frac", "cross_acc", "true_acc", "diff"]:
            rs_string += f"h_{plot_generator.git_hash}_timeout_precalc_{param}_rt_weight_{weight_string}_temp_{temp_string} = np.array(["
            for val in avg_timeout_results[param]:
                rs_string += f"{val}, "
            rs_string += "])\n"
    else:
        avg_timeout_results = None

    avg_instance_selection_results = {}
    for function_name, results in all_instance_selection_results.items():
        if len(results) == 0:
            continue
        avg_instance_selection_results[function_name] = (
            compute_average_grid(results, grid_size=total_samples)
        )
        for param in ["runtime_frac", "cross_acc", "true_acc", "diff"]:
            rs_string += f"h_{plot_generator.git_hash}_{function_name}_{param}_rt_weight_{weight_string}_temp_{temp_string} = np.array(["
            for val in avg_instance_selection_results[function_name][param]:
                rs_string += f"{val}, "
            rs_string += "])\n"

    plot_generator.plot_avg_results(
        avg_timeout_results, avg_instance_selection_results
    )

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
            plot_generator = PlotGenerator(git_hash)
        else:
            if temp is None:
                plot_generator = PlotGenerator(git_hash, f"rt_weight_{rt_weight}")
            else:
                plot_generator = PlotGenerator(git_hash, f"rt_weight_{rt_weight}_temp_{temp}")
                print(f"running with a temperature of {temp}")
        print(f"running with a runtime weight of {rt_weight}")
        # reset results
        global all_timeout_results
        all_timeout_results = []
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

    print(f"sample result after {sample_result_after_iterations} iterations")

    # min value of df > 0.0 is 0.003021
    # clean the 3 runtimes of 0.0s
    df = df.replace(0.0, 0.0001)

    df_runtimes = df.replace([np.inf, -np.inf], 5000)

    df_rated = df.replace([np.inf, -np.inf], 10000)

    par_2_scores_series = df_rated.mean(axis=0)

    results = par_2_scores_series.reset_index()
    results.columns = ['SolverName', 'Par2Score']

    results['Par2Diff'] = None
    results['CrossAcc'] = None
    results['TrueAcc'] = None
    results['RuntimeFrac'] = None

    # initialize global result dict with empty lists
    for instance_selection in experiment_config.instance_selections:
        if instance_selection.__name__ == "no_selection":
            continue
        all_instance_selection_results[instance_selection.__name__] = []

    random_solver_order = list(range(28))

    random.shuffle(random_solver_order)

    for index, solver_index in enumerate(random_solver_order):
        solver_string = f"{solver_index}_" + \
                        f"{par_2_scores_series.index[solver_index]}"

        if index >= break_after_solvers:
            break

        print(f"removing solver {par_2_scores_series.index[solver_index]}")

        # remove solver from par-2-scores and runtimes and build fast C arrays
        reduced_par_2_scores_series = par_2_scores_series.drop(
            par_2_scores_series.index[solver_index]
        )
        par_2_score_removed_solver = par_2_scores_series[solver_index]
        runtime_of_removed_solver = np.ascontiguousarray(
            np.asarray(df_runtimes.iloc[:, solver_index].values), dtype=np.float32
        )
        par_2_scores = np.ascontiguousarray(
            np.asarray(reduced_par_2_scores_series.values), dtype=np.float32
        )
        mean_par_2_score = par_2_scores.mean()
        reduced_df_runtimes = df_runtimes.drop(
            df_runtimes.columns[solver_index], axis=1
        )
        total_runtime = reduced_df_runtimes.stack().sum()
        sorted_runtimes = convert_to_sorted_runtimes(reduced_df_runtimes)

        acc_calculator = Accuracy(
            total_runtime, break_after_runtime_fraction,
            sample_result_after_iterations,
            sorted_runtimes, par_2_scores, mean_par_2_score,
            par_2_score_removed_solver, runtime_of_removed_solver,
            experiment_config.select_idx,
            rt_weight
        )

        # determine thresholds for perfect differentiation of remaining solvers
        thresholds = experiment_config.determine_thresholds(
            acc_calculator, solver_string
        )

        print("here is the calculated threshold vector:")

        for threshold in thresholds:
            print(f"{threshold}", end=", ")
        print()

        print(f"adding solver {par_2_scores_series.index[solver_index]} back in")

        print("here are the results")

        last_results = acc_calculator.sample_result(thresholds, acc_calculator.pred)

        results.iloc[
            solver_index, results.columns.get_loc('Par2Diff')
        ] = last_results["diff"]
        results.iloc[
            solver_index, results.columns.get_loc('CrossAcc')
        ] = last_results["cross_acc"]
        results.iloc[
            solver_index, results.columns.get_loc('TrueAcc')
        ] = last_results["true_acc"]
        results.iloc[
            solver_index, results.columns.get_loc('RuntimeFrac')
        ] = last_results["runtime_frac"]

        print(results)

        print("this gives a mean of")

        print(results.mean())

        solver_results = pd.DataFrame(acc_calculator.solver_results)
        all_timeout_results.append(solver_results)

        for instance_selection in experiment_config.instance_selections:
            print(f"select instances based on method {instance_selection.__name__}")
            if instance_selection.__name__ == "no_selection":
                continue

            selector = InstanceSelector(
                thresholds, sorted_runtimes, acc_calculator,
                sample_result_after_instances, instance_selection
            )

            selector.make_selection()
            selection_results = pd.DataFrame(selector.results)
            all_instance_selection_results[instance_selection.__name__].append(
                selection_results
            )
        if experiment_config.individual_solver_plots:
            plot_generator.plot_solver_results(
                solver_results,
                all_instance_selection_results,
                solver_string
            )
    print(f"length of all_timeout_results is {len(all_timeout_results)}")

    return store_and_get_mean_result(rt_weight, temp)


if __name__ == "__main__":

    git_hash = get_git_commit_hash()

    plot_generator = PlotGenerator(git_hash)
    plot_generator.create_progress_plot()

    print(f"start experiment on {git_hash}")
    run_multi_temp_experiments(experiment_configs)

    print("ended experiment")
