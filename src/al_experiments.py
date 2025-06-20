import pandas as pd
import time
import random
import pickle
import subprocess
from al_experiments.experiment_config import ExperimentConfig
from al_experiments.accuracy import Accuracy
from scipy.interpolate import interp1d

from al_experiments.plot_generator import PlotGenerator
from al_experiments.instance_selector import InstanceSelector, choose_instances_random, variance_based_selection_1, variance_based_selection_2

useCupy = True

if useCupy:
    import numpy
    import cupy as np
else:
    import numpy as np

# constants
number_of_solvers = 28
solver_fraction = 1/number_of_solvers
square_of_solvers = number_of_solvers * number_of_solvers
reduced_square_of_solvers = number_of_solvers*(number_of_solvers-1)
number_of_instances = 5355
# global config
break_after_solvers = 200
break_after_runtime_fraction = 0.655504  # determined by 0e993e00
total_samples = 500  # max is 5354 because of sample_result_after_instances
sample_result_after_iterations = int(number_of_instances * (number_of_solvers - 1) / total_samples)
sample_result_after_instances = int(number_of_instances / total_samples)
# total_runtime = 25860323 s
# global results
all_timeout_results = []
all_random_sel_results = []
all_var_sel_results = []
all_var_sel_2_results = []


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
    data = runtimes.values

    sorted_runtimes_list = []

    for row in data:
        # get sorted indices based on runtime
        sorted_idx = np.argsort(row)
        # create list of (solver_index, runtime) tuples
        tuples = [(int(i), float(row[i])) for i in sorted_idx]
        # add zero element
        tuples.insert(0, (-1, 0))
        sorted_runtimes_list.append(tuples)

    dtype = [('idx', np.int64), ('runtime', np.float64)]
    # 1) allocate a (n_runs, L) structured array
    sorted_rt = np.empty((number_of_instances, number_of_solvers), dtype=dtype)

    # 2) fill it in
    for i in range(number_of_instances):
        sorted_rt[i, :] = sorted_runtimes_list[i]

    return sorted_rt


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
        thresholds: np.ndarray[np.floating[np.float32]],
        runtimes: np.ndarray[np.floating[np.float32]],
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
        thresholds: np.ndarray[np.floating[np.float32]],
        runtimes: np.ndarray[np.floating[np.float32]],
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


def store_and_show_mean_result():

    if not all_timeout_results[0].empty:
        avg_timeout_results = compute_average_grid(all_timeout_results, grid_size=total_samples)
    else:
        avg_timeout_results = None
    avg_random_sel_results = compute_average_grid(all_random_sel_results, grid_size=total_samples)
    avg_var_sel_results = compute_average_grid(all_var_sel_results, grid_size=total_samples)
    avg_var_sel_2_results = compute_average_grid(all_var_sel_2_results, grid_size=total_samples)

    pd.set_option('display.max_rows', total_samples * 2)
    print("avg_timeout_results")
    print(avg_timeout_results)
    print("avg_random_sel_results")
    print(avg_random_sel_results)
    print("avg_var_sel_results")
    print(avg_var_sel_results)
    print("avg_var_sel_2_results")
    print(avg_var_sel_2_results)
    pd.reset_option("display.max_rows")
    plot_generator.plot_avg_results(
        avg_timeout_results, avg_random_sel_results,
        avg_var_sel_results, avg_var_sel_2_results
    )


def static_timeout(
        acc_calculator: Accuracy,
        solver_string: str,
) -> np.ndarray[np.floating[np.float32]]:
    return np.ascontiguousarray(
        np.full((5355,), 27, dtype=np.int32)
    )


def quantized_min_diff(
        acc_calculator: Accuracy,
        solver_string: str,
) -> np.ndarray[np.floating[np.float32]]:

    # initialize tresholds with 0
    thresholds = np.ascontiguousarray(
        np.full((5355,), 0, dtype=np.int32)
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


def run_experiment(experiment_config: ExperimentConfig):
    with open(
        "../al-for-sat-solver-benchmarking-data/pickled-data/anni_full_df.pkl",
        "rb"
    ) as file:
        df: pd.DataFrame = pickle.load(file).copy()

    print(df)

    print(f"sample result after {sample_result_after_iterations} iterations")

    df_runtimes = df.replace([np.inf, -np.inf], 5000)

    df_rated = df.replace([np.inf, -np.inf], 10000)

    par_2_scores_series = df_rated.mean(axis=0)

    results = par_2_scores_series.reset_index()
    results.columns = ['SolverName', 'Par2Score']

    results['Par2Diff'] = None
    results['CrossAcc'] = None
    results['TrueAcc'] = None
    results['RuntimeFrac'] = None

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
            df_runtimes.iloc[:, solver_index], dtype=np.float32
        )
        par_2_scores = np.ascontiguousarray(
            reduced_par_2_scores_series, dtype=np.float32
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
            par_2_score_removed_solver, runtime_of_removed_solver
        )

        # determine thresholds for perfect differentiation of remaining solvers
        thresholds = experiment_config.determine_thresholds(
            acc_calculator, solver_string
        )

        print("here is the calculated threshold vector:")

        for threshold in thresholds:
            print(f"{threshold}", end=", ")

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

        print(f"took {0} calculation steps")

        random_selector = InstanceSelector(
            thresholds, sorted_runtimes, acc_calculator,
            sample_result_after_instances, choose_instances_random
        )

        variance_based_selector = InstanceSelector(
            thresholds, sorted_runtimes, acc_calculator,
            sample_result_after_instances, variance_based_selection_1
        )

        variance_based_selector_2 = InstanceSelector(
            thresholds, sorted_runtimes, acc_calculator,
            sample_result_after_instances, variance_based_selection_2
        )

        random_selector.make_selection()
        variance_based_selector.make_selection()
        variance_based_selector_2.make_selection()

        random_selection_results = random_selector.results
        variance_selection_results = variance_based_selector.results
        variance_selection_2_results = variance_based_selector_2.results

        solver_results = acc_calculator.solver_results

        """  if len(solver_results) > 0:
            del solver_results[0] """

        solver_results = pd.DataFrame(solver_results)
        random_selection_results = pd.DataFrame(random_selection_results)
        variance_selection_results = pd.DataFrame(variance_selection_results)
        variance_selection_2_results = pd.DataFrame(variance_selection_2_results)

        all_timeout_results.append(solver_results)
        all_random_sel_results.append(random_selection_results)
        all_var_sel_results.append(variance_selection_results)
        all_var_sel_2_results.append(variance_selection_2_results)

        plot_generator.plot_solver_results(
            solver_results, random_selection_results, variance_selection_results, variance_selection_2_results, solver_string
        )

    store_and_show_mean_result()


if __name__ == "__main__":

    git_hash = get_git_commit_hash()

    plot_generator = PlotGenerator(git_hash)
    #plot_generator.create_progress_plot()

    # experiment config
    experiment_config = ExperimentConfig(quantized_min_diff)

    print(f"start experiment on {git_hash}")

    run_experiment(experiment_config)

    print("ended experiment")
