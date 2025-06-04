import numpy as np
import pandas as pd
import logging
import sys
import time
import random
import os
import pickle
import warnings

from typing import List, Tuple
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
import matplotlib.pyplot as plt

from al_experiments.config import DEBUG
from al_experiments.clustering import merge_closest_intervals, assign_labels_from_label_boundaries
from al_experiments.evaluation import (
    total_amount_runtime_target_solver,
    mean_wilcoxon_pvalue_predicted_labels,
)
from al_experiments.final_experiments import all_experiments
from al_experiments.experiment import Experiment
from al_experiments.helper import push_notification
from al_experiments.accuracy import accuracy
from scipy.interpolate import interp1d
from IPython.display import display

# constants
number_of_solvers = 28
solver_fraction = 1/number_of_solvers
square_of_solvers = number_of_solvers * number_of_solvers
reduced_square_of_solvers = number_of_solvers*(number_of_solvers-1)
number_of_instances = 5355
# global results
result_tracker = []


logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
    level=logging.INFO,
    handlers=[
        logging.FileHandler("./run.log"),
        logging.StreamHandler(sys.stdout)
    ]
)


def run_e2e_experiment(
    i: int, experiment: Experiment, target_solver: str
) -> Tuple[Tuple[int, int, float, float, float, float], np.ndarray]:
    """ Runs a single end-to-end experiment.

    Args:
        i (int): The number of the experiment.
        experiment (Experiment): The experiment configuration.
        target_solver (str): The target solver.

    Returns:
        Tuple[int, int, float, float, float, float]: The performance summary.
    """

    # Features
    with open("../al-for-sat-solver-benchmarking-data/pickled-data/base_features_df.pkl", "rb") as file:
        base_features_df: pd.DataFrame = pickle.load(file).copy()

    if experiment.only_hashes:
        # Load full runtimes of 16 solvers and filter to competition instances
        with open("../al-for-sat-solver-benchmarking-data/pickled-data/runtimes_df.pkl", "rb") as file:
            runtimes_df: pd.DataFrame = pickle.load(file).copy()
        with open(experiment.instance_filter, "rb") as file:
            instance_filter: pd.Series = pickle.load(file).copy()
    else:
        # Load dedicated competition dataset with more solvers
        with open(f"../al-for-sat-solver-benchmarking-data/pickled-data/{experiment.instance_filter_prefix}_df.pkl", "rb") as file:
            runtimes_df = pickle.load(file).copy()
        instance_filter = runtimes_df.index.copy()

    # Notify of experiment start
    len_solvers = len(list(runtimes_df.columns))
    if DEBUG:
        logging.info(
            f'({i+1}/{len_solvers}) - (0/2) Starting experiment '
            f'"{experiment.key}" for target solver "{target_solver}".'
        )

    # Filter instances
    base_features_df = base_features_df.loc[instance_filter].copy()
    runtimes_df = runtimes_df.loc[instance_filter].copy()
    if not np.all(base_features_df.index == instance_filter) \
            or not np.all(runtimes_df.index == instance_filter):
        logging.error("Filtering instances failed.")

    # Feature log-normalization
    lognorm_base_features_df: pd.DataFrame = (
        (np.log1p(base_features_df) - np.log1p(base_features_df).mean())
        / np.log1p(base_features_df).std()
    ).iloc[:, :-1].copy()

    # Train and target solver
    solvers = list(runtimes_df.columns)
    num_test_set_solvers = 0  # hold-out some solvers
    test_solvers = list(np.random.choice(
        solvers, size=num_test_set_solvers, replace=False))
    train_solvers = solvers.copy()
    for s in test_solvers:
        train_solvers.remove(s)
    train_solvers.remove(target_solver)

    # Cluster based on solvers in training set
    clustering, cluster_boundaries = merge_closest_intervals(
        train_solvers, runtimes_df)
    target_labels = assign_labels_from_label_boundaries(
        cluster_boundaries, runtimes_df, target_solver,
    )
    clustering = clustering.copy()  # Reduces fragmentation

    # Log-normalize data based on only solvers in training set
    lognorm_runtimes_sep_timeout_df = runtimes_df[train_solvers].replace(
        [-np.inf, np.inf, np.nan], 0).astype(dtype=np.int8).copy()

    for idx, (_, instance) in enumerate(list(runtimes_df[train_solvers].iterrows())):
        if not np.all(np.isinf(instance)):
            sorted_rt = np.sort(instance.replace([np.inf], np.nan).dropna())
            if sorted_rt.shape[0] != 1 and np.unique(sorted_rt).shape[0] > 1:
                log_sorted = np.log1p(sorted_rt)
                lognormal_sorted = (
                    log_sorted - np.mean(log_sorted)) / np.std(log_sorted)
            else:
                lognormal_sorted = np.array([0.0])
            transform_map = {orig_val: trans_val for orig_val,
                             trans_val in zip(sorted_rt, lognormal_sorted)}
            transform_map[np.inf] = 0.0
        else:
            transform_map = {np.inf: 0.0}

        for solver_idx, runtime in enumerate(instance):
            lognorm_runtimes_sep_timeout_df.iloc[idx, solver_idx] = float(
                transform_map[runtime])

    for col in train_solvers:
        # Add is_timeout column for every solver
        lognorm_runtimes_sep_timeout_df[f"{col}_istimeout"] = np.where(
            np.isinf(runtimes_df[col]), 1, 0)

    # Base and runtime data
    features = pd.concat([
        lognorm_base_features_df,
        lognorm_runtimes_sep_timeout_df
    ], axis=1)
    features = features.copy()  # Reduce fragmentation

    # Add one-hot encoded clustering data
    for solver in train_solvers:
        for lbl in range(2):
            features[f"{solver}_is_{lbl}"] = np.where(
                clustering[solver] == lbl, 1, 0)

    # True Ranking for Evaluation (binary vector indicating against which solvers it is better)
    other_sampled_runtimes: np.ndarray = np.mean(runtimes_df.loc[
        :, runtimes_df.columns != target_solver
    ].replace([np.inf], experiment.par * 5000), axis=0).to_numpy()
    target_sampled_runtimes: np.ndarray = np.mean(runtimes_df.loc[
        :, runtimes_df.columns == target_solver
    ].replace([np.inf], experiment.par * 5000), axis=0).to_numpy()
    true_par2_ranking = other_sampled_runtimes <= target_sampled_runtimes

    # True label-induced ranking
    other_cluster_labels = clustering.loc[
        :, clustering.columns != target_solver
    ]
    other_label_scores: np.ndarray = np.mean(np.where(
        other_cluster_labels == 2,
        2 * experiment.label_ranking_timeout_weight,
        other_cluster_labels
    ), axis=0)
    target_label_score: float = np.mean(np.where(
        target_labels == 2,
        2 * experiment.label_ranking_timeout_weight,
        target_labels
    ))
    true_label_ranking = other_label_scores <= target_label_score

    # True ranking score
    true_par2_runtimes: np.ndarray = np.mean(
        runtimes_df.replace([np.inf], experiment.par * 5000), axis=0
    ).to_numpy()
    true_par2_runtimes_list = enumerate(list(true_par2_runtimes))
    true_par2_runtimes = np.array(
        list(sorted(true_par2_runtimes_list, key=lambda x: (x[1], x[0]))))

    # Remove all columns belonging to the target solver
    # (Avoid data leakage)
    x: np.ndarray = features.loc[:, (
        (features.columns != f"{target_solver}") &
        (features.columns != f"{target_solver}_istimeout") &
        (features.columns != f"{target_solver}_is_0") &
        (features.columns != f"{target_solver}_is_1")
    )].to_numpy()
    y = target_labels
    y_sampled = np.zeros_like(y)

    # Two-level model for discrete runtime prediction
    # Fixed timeout prediction model
    timeout_clf = StackingClassifier(
        estimators=[
            ("qda", QuadraticDiscriminantAnalysis(reg_param=0, tol=0)),
            ("rf", RandomForestClassifier(
                criterion="entropy", class_weight="balanced")),
        ],
        final_estimator=DecisionTreeClassifier(
            criterion="gini", splitter="best", max_depth=5)
    )

    # Discrete label prediction
    label_clf = StackingClassifier(
        estimators=[
            ("qda", QuadraticDiscriminantAnalysis(reg_param=0, tol=0)),
            ("rf", RandomForestClassifier(criterion="entropy", class_weight="balanced"))
        ],
        final_estimator=DecisionTreeClassifier(
            criterion="gini", splitter="best", max_depth=5)
    )

    # Keep history
    pred_history: List[np.ndarray] = []
    ranking_history: List[np.ndarray] = []
    wilcoxon_history: List[float] = []

    # Add samples until stopping criterion reached
    logging.info(
        f'({i+1}/{len_solvers}) - (1/2) Entering main AL '
        f'loop of experiment "{experiment.key}" '
        f'for target solver "{target_solver}".'
    )
    while not experiment.stopping(
        y_sampled, runtimes_df, clustering, cluster_boundaries,
        timeout_clf, label_clf, x, y, target_solver, experiment,
        ranking_history, wilcoxon_history,
    ) and np.count_nonzero(y_sampled) < y_sampled.shape[0]:
        # Select sample
        sampled_instances_before = int(np.count_nonzero(y_sampled))
        experiment.selection(
            y_sampled, runtimes_df, clustering, cluster_boundaries,
            timeout_clf, label_clf, x, y, target_solver, experiment
        )
        if sampled_instances_before >= int(np.count_nonzero(y_sampled)):
            logging.error(
                f'({i+1}/{len_solvers}) - (FATAL) Sampling did not '
                f'succeed to select next instance in experiment '
                f'"{experiment.key}" for target solver "{target_solver}".'
            )
            break

        # Update p
        p = np.count_nonzero(y_sampled) / y_sampled.shape[0]

        # Split which data is marked selected and which not
        x_train: np.ndarray = x[y_sampled == 1]
        x_test: np.ndarray = x[y_sampled == 0]
        y_train: np.ndarray = y[y_sampled == 1]
        y_test: np.ndarray = y[y_sampled == 0]

        # Permute sampled data
        train_perm = np.random.permutation(x_train.shape[0])
        x_train = x_train[train_perm].copy()
        y_train = y_train[train_perm].copy()
        test_perm = np.random.permutation(x_test.shape[0])
        x_test = x_test[test_perm].copy()
        y_test = y_test[test_perm].copy()

        # Separate training data for each level of the model
        y_train_timeout = np.where(y_train == 2, 1, 0)
        non_timeout = (y_train != 2)
        x_train_non_timeout = x_train[non_timeout]
        y_train_non_timeout = y_train[non_timeout]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Train timeout predictor
                timeout_clf.fit(x_train, y_train_timeout)

                # Train discrete label predictor
                label_clf.fit(x_train_non_timeout, y_train_non_timeout)
        except ValueError:
            if not experiment.ranking.needs_predictions:
                current_ranking = experiment.ranking(
                    y_sampled, runtimes_df, clustering, cluster_boundaries,
                    timeout_clf, label_clf, x, y, target_solver,
                    np.array([]), experiment, pred_history
                )
                ranking_history.append(current_ranking)
                amount_correct_par2_ranking = max(
                    0, (
                        np.count_nonzero(current_ranking == true_par2_ranking) /
                        true_par2_ranking.shape[0]
                    )
                )
                amount_correct_label_ranking = max(
                    0, (
                        np.count_nonzero(current_ranking == true_label_ranking) /
                        true_label_ranking.shape[0]
                    )
                )
            else:
                amount_correct_par2_ranking = 0.0
                amount_correct_label_ranking = 0.0

            perf_tuple = (
                i,
                np.count_nonzero(y_sampled),
                total_amount_runtime_target_solver(
                    y_sampled, runtimes_df, cluster_boundaries,
                    target_solver, experiment,
                ),
                amount_correct_par2_ranking,
                amount_correct_label_ranking,
                0.0,
            )
            continue

        # Predict test set
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            is_timeout_pred = timeout_clf.predict(x)
            label_pred = label_clf.predict(x)
        pred = np.where(is_timeout_pred == 1, 2, label_pred)
        pred_history.append(pred)

        # Ranking evaluation
        current_ranking = experiment.ranking(
            y_sampled, runtimes_df, clustering, cluster_boundaries,
            timeout_clf, label_clf, x, y, target_solver, pred,
            experiment, pred_history,
        )
        ranking_history.append(current_ranking)
        amount_correct_par2_ranking = max(
            0, (
                np.count_nonzero(current_ranking == true_par2_ranking) /
                true_par2_ranking.shape[0]
            )
        )
        amount_correct_label_ranking = max(
            0, (
                np.count_nonzero(current_ranking == true_label_ranking) /
                true_label_ranking.shape[0]
            )
        )

        w = mean_wilcoxon_pvalue_predicted_labels(
            clustering, pred, y, y_sampled, target_solver, experiment
        )
        wilcoxon_history.append(w)

        # Spearman correlation
        current_ranking_score: float = np.mean(np.where(
            pred == 2,
            2 * experiment.label_ranking_timeout_weight,
            pred,
        ))
        current_ranking_scores = np.zeros_like(true_par2_runtimes[:, 1])
        for j in range(len_solvers):
            scores_idx = np.where(true_par2_runtimes[:, 0] == j)[0][0]
            if j == i:  # Target solver
                current_ranking_scores[scores_idx] = current_ranking_score
            elif j > i:
                current_ranking_scores[scores_idx] = other_label_scores[j - 1]
            else:
                current_ranking_scores[scores_idx] = other_label_scores[j]
        spearman_correlation = spearmanr(
            true_par2_runtimes[:, 1], current_ranking_scores)[0]

        perf_tuple = (
            i,
            np.count_nonzero(y_sampled),
            total_amount_runtime_target_solver(
                y_sampled, runtimes_df, cluster_boundaries,
                target_solver, experiment,
            ),
            amount_correct_par2_ranking,
            amount_correct_label_ranking,
            spearman_correlation,
        )
        if DEBUG:
            logging.info((
                "Solver: {}, "
                "Number Instances: {}, "
                "Amount Runtime: {:.6f}, "
                "Accuracy PAR2 Ranking: {:.6f}, "
                "Accuracy Label Ranking: {:.6f}, "
                "Spearman correlation: {:.6f}"
            ).format(*perf_tuple))

    logging.info(
        f'({i+1}/{len_solvers}) - (2/2) Finished AL '
        f'experiment "{experiment.key}" '
        f'for target solver "{target_solver}".'
    )

    return (perf_tuple, y_sampled)


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
    thresholds: 1D array‑like of shape (5355,)
    runtimes:    2D array‑like of shape (5355, 28)

    For each i, any runtimes[i, j] > thresholds[i] is replaced by thresholds[i],
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
    thresholds: 1D array‑like of shape (5355,)
    runtimes:    2D array‑like of shape (5355, 28)

    For each i, any runtimes[i, j] > thresholds[i] is replaced by thresholds[i],
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

    ax1.plot(avg_results["runtime_frac"], avg_results["diff"], 'g-o', label="diff")
    ax1.set_ylabel("diff", color='g')

    ax2.plot(avg_results["runtime_frac"], avg_results["cross_acc"], 'b-s', label="cross_acc")
    ax2.plot(avg_results["runtime_frac"], avg_results["true_acc"], 'r-x', label="true_acc")
    ax2.set_ylabel("cross_acc", color='b')
    ax2.set_ylabel("true_acc", color='r')

    plt.title("average over all solvers")

    # optional: add grids and legends
    ax1.grid(True)
    ax1.set_xlabel("runtime fraction")
    fig.tight_layout()
    fig.savefig("./plots/test-delete/average_results.png", dpi=300)


def determine_tresholds(
        runtime_per_step: float,
        total_runtime: float,
        par_2_scores: np.ndarray[np.floating[np.float32]],
        runtimes: np.ndarray[np.floating[np.float32]],
        acc_calculator: accuracy,
        progress: str,
        solver_string: str,
        solver_index: int,
        par_2_scores_series,
        df_runtimes,
        df_rated,
        rated_runtimes: np.ndarray[np.floating[np.float32]]
) -> np.ndarray[np.floating[np.float32]]:

    max_runtime_per_step = 2 * runtime_per_step
    min_par_2_score = par_2_scores.min()

    print(f"adding {runtime_per_step}s of runtime per step")

    # initialize tresholds with 0
    thresholds = np.ascontiguousarray(
        np.full((5355,), 0), dtype=np.float32
    )
    start = time.time_ns()
    max_acc = 0
    min_diff = 999999999.0

    solver_results = []

    for i in range(calc_steps*100):  # will break earlier (if runtime fracion is reached)
        runtime_to_add = random.random() * max_runtime_per_step
        thresholds, max_acc, min_diff = acc_calculator.add_runtime(
            thresholds, rated_runtimes, par_2_scores,
            min_par_2_score, runtime_to_add, max_acc, min_diff
        )
        if min_diff == -1:
            break
        if i % 20 == 0:
            runtime_frac = vec_to_runtime_frac(
                thresholds, runtimes, total_runtime
            )
            print(f"{progress} runtime fraction is {runtime_frac}")
            solver_results.append(get_stats(df_rated, df_runtimes, par_2_scores_series, par_2_scores, runtimes, thresholds, solver_index, acc_calculator, progress))
            if runtime_frac > 0.018:
                break
    print(f"took {(time.time_ns() - start) / 1_000_000_000}s")

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
    fig.savefig(f"./plots/test-delete/{solver_string}_results.png", dpi=300)

    return thresholds


def get_stats(df_rated, df_runtimes, par_2_scores_series, par_2_scores, runtimes, thresholds, solver_index, acc_calculator: accuracy, progress: str):
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

    """    random_baseline_whole_instances_runtime_frac = np.array([0.012854, 0.014792, 0.016731, 0.018669, 0.020608, 0.022546, 0.024485, 0.026423, 0.028362, 0.030300, 0.032239, 0.034177, 0.036116, 0.038054, 0.039993, 0.041931, 0.043869, 0.045808, 0.047746, 0.049685, 0.051623, 0.053562, 0.055500, 0.057439, 0.059377, 0.061316, 0.063254, 0.065193, 0.067131, 0.069070, 0.071008, 0.072946, 0.074885, 0.076823, 0.078762, 0.080700, 0.082639, 0.084577, 0.086516, 0.088454, 0.090393, 0.092331, 0.094270, 0.096208, 0.098147, 0.100085, 0.102023, 0.103962, 0.105900, 0.107839, 0.109777, 0.111716, 0.113654, 0.115593, 0.117531, 0.119470, 0.121408, 0.123347, 0.125285, 0.127223, 0.129162, 0.131100, 0.133039, 0.134977, 0.136916, 0.138854, 0.140793, 0.142731, 0.144670, 0.146608, 0.148547, 0.150485, 0.152424, 0.154362, 0.156300, 0.158239, 0.160177, 0.162116, 0.164054, 0.165993, 0.167931, 0.169870, 0.171808, 0.173747, 0.175685, 0.177624, 0.179562, 0.181501, 0.183439, 0.185377, 0.187316, 0.189254, 0.191193, 0.193131, 0.195070, 0.197008, 0.198947, 0.200885, 0.202824, 0.204762])   
    random_baseline_whole_instances_true_acc = np.array([0.800671, 0.815994, 0.831123, 0.840869, 0.849788, 0.858210, 0.865402, 0.871996, 0.876478, 0.881804, 0.886714, 0.890490, 0.896291, 0.900608, 0.902859, 0.904178, 0.904763, 0.904025, 0.901968, 0.900711, 0.901171, 0.902492, 0.905383, 0.909716, 0.913746, 0.917232, 0.920133, 0.920813, 0.920997, 0.920607, 0.919150, 0.919176, 0.919301, 0.920650, 0.923952, 0.927819, 0.931296, 0.933643, 0.935445, 0.936112, 0.935805, 0.934204, 0.931470, 0.928752, 0.925632, 0.923231, 0.921840, 0.920926, 0.921713, 0.922844, 0.925048, 0.927268, 0.928746, 0.929591, 0.929659, 0.929496, 0.929816, 0.929342, 0.928226, 0.927260, 0.927368, 0.928662, 0.930807, 0.932590, 0.933650, 0.934468, 0.934882, 0.934914, 0.934747, 0.934176, 0.932449, 0.930397, 0.928821, 0.928179, 0.928629, 0.929744, 0.932620, 0.936301, 0.940253, 0.943307, 0.945378, 0.946918, 0.947236, 0.946021, 0.945092, 0.943900, 0.943051, 0.942690, 0.942132, 0.941903, 0.942127, 0.942416, 0.942220, 0.941933, 0.941853, 0.941881, 0.942131, 0.943495, 0.945489, 0.946260])

    random_baseline_dynamic_timeout_runtime_frac = np.array([0.000233, 0.002042, 0.003850, 0.005658, 0.007467, 0.009275, 0.011083, 0.012891, 0.014700, 0.016508, 0.018316, 0.020125, 0.021933, 0.023741, 0.025550, 0.027358, 0.029166, 0.030974, 0.032783, 0.034591, 0.036399, 0.038208, 0.040016, 0.041824, 0.043633, 0.045441, 0.047249, 0.049057, 0.050866, 0.052674, 0.054482, 0.056291, 0.058099, 0.059907, 0.061716, 0.063524, 0.065332, 0.067140, 0.068949, 0.070757, 0.072565, 0.074374, 0.076182, 0.077990, 0.079799, 0.081607, 0.083415, 0.085223, 0.087032, 0.088840, 0.090648, 0.092457, 0.094265, 0.096073, 0.097882, 0.099690, 0.101498, 0.103306, 0.105115, 0.106923, 0.108731, 0.110540, 0.112348, 0.114156, 0.115965, 0.117773, 0.119581, 0.121389, 0.123198, 0.125006, 0.126814, 0.128623, 0.130431, 0.132239, 0.134048, 0.135856, 0.137664, 0.139472, 0.141281, 0.143089, 0.144897, 0.146706, 0.148514, 0.150322, 0.152131, 0.153939, 0.155747, 0.157555, 0.159364, 0.161172, 0.162980, 0.164789, 0.166597, 0.168405, 0.170214, 0.172022, 0.173830, 0.175639, 0.177447, 0.179255])    
    random_baseline_dynamic_timeout_true_acc = np.array([0.732619, 0.856523, 0.868901, 0.882169, 0.883973, 0.879346, 0.891275, 0.891847, 0.894557, 0.893746, 0.896611, 0.890922, 0.901432, 0.908260, 0.902582, 0.907933, 0.902097, 0.899038, 0.903821, 0.906894, 0.902879, 0.908655, 0.909097, 0.908730, 0.907207, 0.907966, 0.913631, 0.907616, 0.909618, 0.910934, 0.907407, 0.903917, 0.904092, 0.898227, 0.900781, 0.906085, 0.903549, 0.907407, 0.898499, 0.898975, 0.908635, 0.903020, 0.900988, 0.906085, 0.907647, 0.908730, 0.911376, 0.901478, 0.901609, 0.904755, 0.908892, 0.903957, 0.899471, 0.902021, 0.899832, 0.903703, 0.903439, 0.903439, 0.904762, 0.912901, 0.915254, 0.914959, 0.911376, 0.915344, 0.918822, 0.915219, 0.914615, 0.914226, 0.915344, 0.916667, 0.915344, 0.919312, 0.921950, 0.916537, 0.917989, 0.914153, 0.924477, 0.927364, 0.916797, 0.918372, 0.919452, 0.923439, 0.923394, 0.929894, 0.927249, 0.925926, 0.929395, 0.928561, 0.928181, 0.924603, 0.932270, 0.932540, 0.933759, 0.931217, 0.935059, 0.935185, 0.933862, 0.937831, 0.933847, 0.931217])

    variance_based_selection_runtime_frac = np.array([0.002298, 0.003724, 0.005151, 0.006578, 0.008004, 0.009431, 0.010858, 0.012284, 0.013711, 0.015138, 0.016564, 0.017991, 0.019418, 0.020844, 0.022271, 0.023698, 0.025124, 0.026551, 0.027978, 0.029404, 0.030831, 0.032258, 0.033685, 0.035111, 0.036538, 0.037965, 0.039391, 0.040818, 0.042245, 0.043671, 0.045098, 0.046525, 0.047951, 0.049378, 0.050805, 0.052231, 0.053658, 0.055085, 0.056511, 0.057938, 0.059365, 0.060791, 0.062218, 0.063645, 0.065071, 0.066498, 0.067925, 0.069351, 0.070778, 0.072205, 0.073631, 0.075058, 0.076485, 0.077911, 0.079338, 0.080765, 0.082192, 0.083618, 0.085045, 0.086472, 0.087898, 0.089325, 0.090752, 0.092178, 0.093605, 0.095032, 0.096458, 0.097885, 0.099312, 0.100738, 0.102165, 0.103592, 0.105018, 0.106445, 0.107872, 0.109298, 0.110725, 0.112152, 0.113578, 0.115005, 0.116432, 0.117858, 0.119285, 0.120712, 0.122138, 0.123565, 0.124992, 0.126418, 0.127845, 0.129272, 0.130699, 0.132125, 0.133552, 0.134979, 0.136405, 0.137832, 0.139259, 0.140685, 0.142112, 0.143539])  
    variance_based_selection_true_acc = np.array([0.802179, 0.815325, 0.821714, 0.829112, 0.837802, 0.840405, 0.844817, 0.848778, 0.849851, 0.850670, 0.851558, 0.852389, 0.854510, 0.859935, 0.867421, 0.874355, 0.878819, 0.880730, 0.882847, 0.884688, 0.886474, 0.889573, 0.891923, 0.893214, 0.894246, 0.895334, 0.896721, 0.898168, 0.899581, 0.901787, 0.904594, 0.907746, 0.910952, 0.913655, 0.914495, 0.914601, 0.913331, 0.912274, 0.912282, 0.912549, 0.913679, 0.914232, 0.913793, 0.913313, 0.912863, 0.914563, 0.915963, 0.916391, 0.916623, 0.916497, 0.916864, 0.917380, 0.917660, 0.918070, 0.918479, 0.919161, 0.919485, 0.920138, 0.920728, 0.920082, 0.919787, 0.919767, 0.920038, 0.921426, 0.923065, 0.924713, 0.926290, 0.928292, 0.929882, 0.931300, 0.933537, 0.935675, 0.937343, 0.938549, 0.939656, 0.940860, 0.942117, 0.943115, 0.944101, 0.945142, 0.945847, 0.946553, 0.947267, 0.947982, 0.948820, 0.949580, 0.950248, 0.950471, 0.950756, 0.950920, 0.950967, 0.950896, 0.950823, 0.951183, 0.951512, 0.951840, 0.952168, 0.952497, 0.952472, 0.952381]) 

    dynamic_timeout_optimized_runtime_frac = np.array([0.001427, 0.002411, 0.003394, 0.004377, 0.005361, 0.006344, 0.007327, 0.008311, 0.009294, 0.010277, 0.011261, 0.012244, 0.013227, 0.014211, 0.015194, 0.016178, 0.017161, 0.018144, 0.019128, 0.020111, 0.021094, 0.022078, 0.023061, 0.024044, 0.025028, 0.026011, 0.026994, 0.027978, 0.028961, 0.029944, 0.030928, 0.031911, 0.032894, 0.033878, 0.034861, 0.035845, 0.036828, 0.037811, 0.038795, 0.039778, 0.040761, 0.041745, 0.042728, 0.043711, 0.044695, 0.045678, 0.046661, 0.047645, 0.048628, 0.049611, 0.050595, 0.051578, 0.052561, 0.053545, 0.054528, 0.055512, 0.056495, 0.057478, 0.058462, 0.059445, 0.060428, 0.061412, 0.062395, 0.063378, 0.064362, 0.065345, 0.066328, 0.067312, 0.068295, 0.069278, 0.070262, 0.071245, 0.072228, 0.073212, 0.074195, 0.075179, 0.076162, 0.077145, 0.078129, 0.079112, 0.080095, 0.081079, 0.082062, 0.083045, 0.084029, 0.085012, 0.085995, 0.086979, 0.087962, 0.088945, 0.089929, 0.090912, 0.091895, 0.092879, 0.093862, 0.094845, 0.095829, 0.096812, 0.097796, 0.098779]) 
    dynamic_timeout_optimized_true_acc = np.array([0.915045, 0.911842, 0.903083, 0.905274, 0.909750, 0.923364, 0.930298, 0.930972, 0.930386, 0.918715, 0.921096, 0.926069, 0.924703, 0.917278, 0.918582, 0.916526, 0.913898, 0.919391, 0.919613, 0.920864, 0.918878, 0.919720, 0.919967, 0.919121, 0.921070, 0.918207, 0.912812, 0.909581, 0.907517, 0.907499, 0.908177, 0.907898, 0.906824, 0.903456, 0.905493, 0.906737, 0.904698, 0.904792, 0.905999, 0.905993, 0.904649, 0.902110, 0.899954, 0.897426, 0.893337, 0.892723, 0.895743, 0.893570, 0.893444, 0.894502, 0.894002, 0.889396, 0.890087, 0.892857, 0.894163, 0.896261, 0.897140, 0.900478, 0.901393, 0.903588, 0.899612, 0.900736, 0.897163, 0.894490, 0.898444, 0.900974, 0.902380, 0.900323, 0.901981, 0.899697, 0.897681, 0.901584, 0.899075, 0.897663, 0.897110, 0.896737, 0.897074, 0.893708, 0.894759, 0.895580, 0.895503, 0.894948, 0.896996, 0.894498, 0.892857, 0.892041, 0.891571, 0.892358, 0.892372, 0.894156, 0.889654, 0.894529, 0.897532, 0.900728, 0.902143, 0.901996, 0.902031, 0.904149, 0.906475, 0.906085]) 

    al_runtime_frac = np.array([0.02106387, 0.02159122, 0.02176197, 0.02385467, 0.02522425, 0.02807994, 0.05256794, 0.14633324])
    al_true_acc = np.array([0.59288538, 0.6798419,  0.69762846, 0.76482213, 0.77865613, 0.80237154, 0.82411067, 0.8972332])

    plt.figure(figsize=(10, 6))
    plt.plot(random_baseline_whole_instances_runtime_frac, random_baseline_whole_instances_true_acc, marker='x', label="random instances")
    plt.plot(random_baseline_dynamic_timeout_runtime_frac, random_baseline_dynamic_timeout_true_acc, marker='x', label="random dynamic timeout")
    plt.plot(variance_based_selection_runtime_frac, variance_based_selection_true_acc, marker='x', label="variance based selection")
    plt.plot(dynamic_timeout_optimized_runtime_frac, dynamic_timeout_optimized_true_acc, marker='x', label="dynamic timeout optimized")
    #plt.plot(al_runtime_frac, al_true_acc, marker='x', label="active learning")
    plt.legend()
    plt.xlabel("Fraction of Runtime")
    plt.xlim(right=0.1)
    plt.ylabel("Ranking Accuracy")
    plt.title("Comparision of different instance selection methods")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    # Show or save
    plt.show()
    #plt.savefig("instance_histogram.png", dpi=300) """

    push_notification("start test")

    acc_calculator = accuracy()

    calc_steps = 1000000
    runtime_per_step = 40
    break_after_solvers = 100
    # total_runtime = 25860323 s

    with open("../al-for-sat-solver-benchmarking-data/pickled-data/anni_full_df.pkl", "rb") as file:
        df: pd.DataFrame = pickle.load(file).copy()

    print(df)


    df_runtimes = df.replace([np.inf, -np.inf], 5000)

    df_rated = df.replace([np.inf, -np.inf], 10000)

    total_runtime = df_runtimes.stack().sum()

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
        solver_string = f"{solver_index}_{par_2_scores_series.index[solver_index]}"

        if index >= break_after_solvers:
            break

        print(f"removing solver {par_2_scores_series.index[solver_index]}")

        # remove solver from par-2-scores and runtimes and build fast C arrays
        reduced_par_2_scores_series = par_2_scores_series.drop(
            par_2_scores_series.index[solver_index]
        )
        par_2_scores = np.ascontiguousarray(
            reduced_par_2_scores_series, dtype=np.float32
        )
        reduced_df_runtimes = df_runtimes.drop(df_runtimes.columns[solver_index], axis=1)
        reduced_df_runtimes_rated = df_rated.drop(df_rated.columns[solver_index], axis=1)
        runtimes = np.ascontiguousarray(
            reduced_df_runtimes.copy(), dtype=np.float32
        )
        rated_runtimes = np.ascontiguousarray(
            reduced_df_runtimes_rated.copy(), dtype=np.float32
        )
        reduced_max_runtime_series = df_runtimes.max(axis=1)
        reduced_max_runtime = np.ascontiguousarray(
            reduced_max_runtime_series.copy(), dtype=np.float32
        )

        # determine thresholds for perfect differentiation of remaining solvers
        thresholds = determine_tresholds(
            runtime_per_step, total_runtime, par_2_scores, runtimes, acc_calculator, f"{index+1}/{random_solver_order.__len__()}", solver_string, solver_index, par_2_scores_series, df_runtimes, df_rated, rated_runtimes
        )

        print("here is the calculated threshold vector:")

        for threshold in thresholds:
            print(f"{threshold}, ")

        print("before adding solver back in, here are the stats:")

        # TODO: this is calculated with the wrong runtimes (it needs the rated ones)
        cross_acc = acc_calculator.vec_to_cross_acc(thresholds, runtimes, par_2_scores)

        acc_calculator.print_key_signature(thresholds, runtimes, par_2_scores)

        #par_2_diff = acc_calculator.vec_to_diff(thresholds, runtimes, par_2_scores, par_2_scores.mean(), index)

        #print(f"difference of both scores is {par_2_diff}")

        print(f"adding solver {par_2_scores_series.index[solver_index]} back in")

        par_2_scores = np.ascontiguousarray(
            par_2_scores_series, dtype=np.float32
        )
        runtimes = np.ascontiguousarray(
            df_runtimes.copy(), dtype=np.float32
        )
        runtimes_rated = np.ascontiguousarray(
            df_rated.copy(), dtype=np.float32
        )

        true_acc = acc_calculator.vec_to_true_acc(
            thresholds, runtimes_rated, par_2_scores, solver_index
        )

        runtime_frac = vec_to_single_runtime_frac(thresholds, runtimes, solver_index)

        print(f"this gives an accuracy (true) of {true_acc}")

        print(f"with a runtime fraction of {runtime_frac} for the new solver")

        print(f"the cross accuracy is {acc_calculator.vec_to_cross_acc(thresholds, runtimes_rated, par_2_scores)}")

        acc_calculator.print_key_signature(thresholds, runtimes, par_2_scores)

        print("inserting into results:")

        results.iloc[
            solver_index, results.columns.get_loc('Par2Diff')
        ] = 0#par_2_diff
        results.iloc[
            solver_index, results.columns.get_loc('CrossAcc')
        ] = cross_acc
        results.iloc[
            solver_index, results.columns.get_loc('TrueAcc')
        ] = true_acc
        results.iloc[
            solver_index, results.columns.get_loc('RuntimeFrac')
        ] = runtime_frac

        print(results)

        print("this gives a mean of")

        print(results.mean())

        print(f"took {calc_steps} calculation steps")

    store_and_show_mean_result()

    """

    # 2a. set infinite value to punishment of 2*tau
    df_non_inf = df.replace([np.inf, -np.inf], 10000)

    par_2_scores = df_non_inf.mean(axis=0, skipna=True)

    average_before_limits = df_non_inf.mean(axis=1, skipna=True)

    runtime_limits = df_non_inf.sum(axis=1, skipna=True) * p

    df_remaining_runtimes = df_non_inf.copy()

    runtime_per_instance = {}
    max_runtime = {}

    # Determine max allowed runtime
    for index, row in df_remaining_runtimes.iterrows():
        runtime_limits[index] = runtime_limits[index] * k/average_before_limits[index]
        runtime_per_instance[index] = 0
        max_runtime[index] = 0
        row_copy = row.sort_values()
        i = 0
        while runtime_per_instance[index] < runtime_limits[index] and i < row_copy.size:
            if i >= 1:
                max_runtime[index] = row_copy.iloc[i-1]
            runtime_per_instance[index] += row_copy.iloc[i]
            i += 1

        row[row < max_runtime[index]] = np.nan

    max_runtime = pd.Series(max_runtime)

    average_after_limits = df_remaining_runtimes.mean(axis=1, skipna=True)

    print("average before limits")
    print(average_before_limits)
    print("average after limits")
    print(average_after_limits)

    df_remaining_runtimes = df_non_inf.copy()
    n_rows = df_remaining_runtimes.shape[0]

    for i in range(n_rows):
        # pull out the i-th row as a Series, map your function, assign it back
        df_remaining_runtimes.iloc[i] = df_remaining_runtimes.iloc[i].map(
            lambda x: x if x < max_runtime[i] else average_after_limits[i]
        )

    predicted_par_2_scores = df_remaining_runtimes.mean(axis=0, skipna=True)

    print("par-2 scores")
    print(par_2_scores.sort_values())
    print("predicted par-2 scores")
    print(predicted_par_2_scores.sort_values())

    determine_acuracy(par_2_scores, predicted_par_2_scores)

    determine_runtime_fraction(df.copy(), max_runtime.copy())


    # 3. Plot the histogram
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

    """ # Notify experiment start
    push_notification("Starting experiments.")

    for i_exp, experiment in enumerate(all_experiments):
        # Skip if present
        time.sleep(random.random() * 5)
        if os.path.exists(f"{experiment.results_location}.csv"):
            continue
        else:
            with open(f"{experiment.results_location}.csv", "w") as lock_file:
                lock_file.write("lock")

        # Retrieve column list
        if experiment.only_hashes:
            with open("../al-for-sat-solver-benchmarking-data/pickled-data/runtimes_df.pkl", "rb") as file:
                runtimes_df = pickle.load(file).copy()
        else:
            with open(f"../al-for-sat-solver-benchmarking-data/pickled-data/{experiment.instance_filter_prefix}_df.pkl", "rb") as file:
                runtimes_df = pickle.load(file).copy()
        column_list = list(runtimes_df.columns)
        del runtimes_df

        # Run cross-validation with each solver as target once
        if DEBUG:
            solver_results = [
                run_e2e_experiment(i, experiment, target_solver)
                for i, target_solver in enumerate(column_list)
            ]
        else:
            solver_results = Parallel(n_jobs=-1)(
                delayed(run_e2e_experiment)(
                    i * experiment.repetitions + j, experiment, target_solver
                )
                for i, target_solver in enumerate(column_list)
                for j in range(experiment.repetitions)
            )

        # Store results
        res_df = pd.DataFrame.from_records(
            [t for t, _ in solver_results],
            columns=[
                "solver", "num_instances", "amount_runtime",
                "par2_ranking_acc", "label_ranking_acc",
                "spearman"
            ],
            index="solver",
        )
        res_df.to_csv(f"{experiment.results_location}.csv")

        # Store sampled instances
        for i, (_, y_sampled) in enumerate(solver_results):
            with open(f"../al-for-sat-solver-benchmarking-data/pickled-data/end_to_end/{experiment.key}_y_sampled_{i:02d}.pkl", "wb") as wfile:
                pickle.dump(y_sampled.copy(), wfile)

        if len(all_experiments) <= 50 or i_exp % 100 == 0:
            push_notification(
                f"{(i_exp+1)}/{len(all_experiments)} = {(i_exp/len(all_experiments)):.2f} ({experiment.key})."
            ) """
