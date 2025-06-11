import numpy as np
import pandas as pd
import time
import random
import pickle
import matplotlib.pyplot as plt
from al_experiments.helper import push_notification
from al_experiments.accuracy import Accuracy
from scipy.interpolate import interp1d

from al_experiments.plot_generator import PlotGenerator

# constants
number_of_solvers = 28
solver_fraction = 1/number_of_solvers
square_of_solvers = number_of_solvers * number_of_solvers
reduced_square_of_solvers = number_of_solvers*(number_of_solvers-1)
number_of_instances = 5355
# config
break_after_solvers = 100
break_after_runtime_fraction = 2
sample_result_after_iterations = 1000

sample_result_after_iterations = 5000
# total_runtime = 25860323 s
# global results
result_tracker = []


def convert_to_sorted_runtimes(runtimes: pd.DataFrame):
    data = runtimes.values

    sorted_runtimes = []
    for row in data:
        # get sorted indices based on runtime
        sorted_idx = np.argsort(row)
        # create list of (solver_index, runtime) tuples
        tuples = [(int(i), float(row[i])) for i in sorted_idx]
        # add zero element
        tuples.insert(0, (-1, 0))
        sorted_runtimes.append(tuples)

    return sorted_runtimes


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


def compute_average_grid(list_of_dfs, grid_size=100):
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
    avg_results = compute_average_grid(result_tracker, grid_size=100)

    pd.set_option('display.max_rows', 110)
    print(avg_results)
    pd.reset_option("display.max_rows")

    # create main plot and a twin y-axis
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()

    ax1.plot(
        avg_results["runtime_frac"],
        avg_results["diff"], 'g-o',
        label="diff"
    )
    ax1.set_ylabel("diff", color='g')

    ax2.plot(
        avg_results["runtime_frac"],
        avg_results["cross_acc"],
        'b-s',
        label="cross_acc"
    )
    ax2.plot(
        avg_results["runtime_frac"],
        avg_results["true_acc"],
        'r-x',
        label="true_acc"
    )
    ax2.set_ylabel("cross_acc", color='b')
    ax2.set_ylabel("true_acc", color='r')

    plt.title("average over all solvers")

    # optional: add grids and legends
    ax1.grid(True)
    ax1.set_xlabel("runtime fraction")
    fig.tight_layout()
    fig.savefig("./plots/quantized runtime min diff to 1/average_results.png", dpi=300)


def determine_tresholds(
        total_runtime: float,
        par_2_scores: np.ndarray[np.floating[np.float32]],
        sorted_runtimes: np.ndarray,
        acc_calculator: Accuracy,
        progress: str,
        solver_string: str,
        solver_index: int,
        par_2_scores_series,
        df_runtimes,
        df_rated,
) -> np.ndarray[np.floating[np.float32]]:

    # initialize tresholds with 0
    thresholds = np.ascontiguousarray(
        np.full((5355,), 0, dtype=np.int32)
    )
    start = time.time_ns()
    max_acc = 0
    min_diff = 999999999.0

    while True:
        thresholds, max_acc, min_diff = acc_calculator.add_runtime_fast(
            thresholds, max_acc, min_diff
        )
        if min_diff == -1:
            break
    print(f"took {(time.time_ns() - start) / 1_000_000_000}s")

    solver_results = acc_calculator.solver_results

    del solver_results[0]
    solver_results = pd.DataFrame(solver_results)

    result_tracker.append(solver_results)

    # create main plot and a twin y-axis
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()

    ax1.plot(solver_results["runtime_frac"], solver_results["diff"], 'g-o', label="diff")
    ax1.set_ylabel("diff", color='g')

    ax2.plot(solver_results["runtime_frac"], solver_results["cross_acc"], 'b-s', label="cross_acc")
    ax2.plot(solver_results["runtime_frac"], solver_results["true_acc"], 'r-x', label="true_acc")
    ax2.set_ylabel("cross_acc", color='b')
    ax2.set_ylabel("true_acc", color='r')

    plt.title(f"dynamic vector for solver {solver_string}")

    # optional: add grids and legends
    ax1.grid(True)
    ax1.set_xlabel("runtime fraction")
    fig.tight_layout()
    fig.savefig(f"./plots/quantized runtime min diff to 1/{solver_string}_results.png", dpi=300)

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


if __name__ == "__main__":

    push_notification("start test")

    plot_generator = PlotGenerator()
    #plot_generator.create_progress_plot()

    with open(
        "../al-for-sat-solver-benchmarking-data/pickled-data/anni_full_df.pkl",
        "rb"
    ) as file:
        df: pd.DataFrame = pickle.load(file).copy()

    print(df)

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
        thresholds = determine_tresholds(
            total_runtime,
            par_2_scores,
            sorted_runtimes,
            acc_calculator,
            f"{index+1}/{random_solver_order.__len__()}",
            solver_string,
            solver_index,
            par_2_scores_series,
            df_runtimes,
            df_rated
        )

        print("here is the calculated threshold vector:")

        for threshold in thresholds:
            print(f"{threshold}", end=", ")

        print(f"adding solver {par_2_scores_series.index[solver_index]} back in")

        print("here are the results")

        last_results = acc_calculator.sample_result(thresholds)

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

    store_and_show_mean_result()

    """     # 3. Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(max_runtime, bins='auto')
    plt.xlabel("total runtime (seconds)")
    plt.ylabel("Count")
    plt.title("Histogram of restricted runtime per SAT instance")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    # Show or save
    plt.show()
    #plt.savefig("instance_histogram.png", dpi=300) """

