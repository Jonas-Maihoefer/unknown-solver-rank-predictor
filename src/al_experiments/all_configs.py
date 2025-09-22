from al_experiments.determine_timeout import create_instance_wise, quantized_mean_punish, quantized_double_punish, static_timeout
from al_experiments.experiment_config import ExperimentConfig, not_needed
from al_experiments.accuracy import Accuracy, create_cross_acc_breaking, create_softmax_fn, create_stability_breaking, create_top_k_sampling, greedy_cross_acc, greedy_rmse, knapsack_cross_acc, knapsack_rmse, select_best_idx
from scipy.interpolate import interp1d

from al_experiments.plot_generator import PlotGenerator
from al_experiments.instance_selector import InstanceSelector, choose_instances_random, highest_variance, variance_based_selection_1, variance_based_selection_2, highest_rt_selection, lowest_variance, lowest_variances_per_rt, lowest_rt_selection
from al_experiments.constants import Constants
from al_experiments.thresh_breaking_condition import ThreshBreakingCondition


experiment_configs = [
    ExperimentConfig(
        filter_unsolvable=True,
        determine_thresholds=quantized_double_punish,  # quantized_double_punish, quantized_mean_punish, create_instance_wise, static_timeout(5000)
        select_idx=create_top_k_sampling(3),
        scoring_fn=greedy_cross_acc,  # knapsack_rmse, greedy_rmse, knapsack_cross_acc, greedy_cross_acc
        thresh_breaking_conditions=[ThreshBreakingCondition('until_stab_0_9800', create_stability_breaking(0.98)), ThreshBreakingCondition('until_stab_0_9825', create_stability_breaking(0.9825)), ThreshBreakingCondition('until_stab_0_9850', create_stability_breaking(0.985)), ThreshBreakingCondition('until_stab_0_9875', create_stability_breaking(0.9875)), ThreshBreakingCondition('until_stab_0_9900', create_stability_breaking(0.99)), ThreshBreakingCondition('until_stab_0_9925', create_stability_breaking(0.9925)), ThreshBreakingCondition('until_stab_0_9950', create_stability_breaking(0.995)), ThreshBreakingCondition('until_stab_0_9975', create_stability_breaking(0.9975))],  # leave one if no breaking cond is needed (instance wise)
        temperatures=[],  # [0.5, 0.35, 0.25, 0.125, 0.09, 0.06125, 0.03075, 0.01530, 0.008, 0.004],
        rt_weights=[1],   # [1.0, 0.95, 1.1, 1.3, 1.5, 0.8, 1.6, 1.2, 1.4, 1.7, 1.05, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0],
        instance_selections=[choose_instances_random, variance_based_selection_1, variance_based_selection_2, highest_rt_selection, lowest_variance, highest_variance, lowest_variances_per_rt, lowest_rt_selection],  # choose_instances_random, variance_based_selection_1, variance_based_selection_2, highest_rt_selection, lowest_variance, highest_variance, lowest_variances_per_rt, lowest_rt_selection
        individual_solver_plots=False
    ),
    ExperimentConfig(
        filter_unsolvable=False,
        determine_thresholds=quantized_double_punish,  # quantized_double_punish, quantized_mean_punish, create_instance_wise, static_timeout(5000)
        select_idx=create_top_k_sampling(3),
        scoring_fn=greedy_rmse,  # knapsack_rmse, greedy_rmse, knapsack_cross_acc, greedy_cross_acc
        thresh_breaking_conditions=[ThreshBreakingCondition('until_stab_0_9800', create_stability_breaking(0.98)), ThreshBreakingCondition('until_stab_0_9825', create_stability_breaking(0.9825)), ThreshBreakingCondition('until_stab_0_9850', create_stability_breaking(0.985)), ThreshBreakingCondition('until_stab_0_9875', create_stability_breaking(0.9875)), ThreshBreakingCondition('until_stab_0_9900', create_stability_breaking(0.99)), ThreshBreakingCondition('until_stab_0_9925', create_stability_breaking(0.9925)), ThreshBreakingCondition('until_stab_0_9950', create_stability_breaking(0.995)), ThreshBreakingCondition('until_stab_0_9975', create_stability_breaking(0.9975))],  # leave one if no breaking cond is needed (instance wise)
        temperatures=[],  # [0.5, 0.35, 0.25, 0.125, 0.09, 0.06125, 0.03075, 0.01530, 0.008, 0.004],
        rt_weights=[1],   # [1.0, 0.95, 1.1, 1.3, 1.5, 0.8, 1.6, 1.2, 1.4, 1.7, 1.05, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0],
        instance_selections=[choose_instances_random, variance_based_selection_1, variance_based_selection_2, highest_rt_selection, lowest_variance, highest_variance, lowest_variances_per_rt, lowest_rt_selection],  # choose_instances_random, variance_based_selection_1, variance_based_selection_2, highest_rt_selection, lowest_variance, highest_variance, lowest_variances_per_rt, lowest_rt_selection
        individual_solver_plots=False
    ),
    ExperimentConfig(
        filter_unsolvable=True,
        determine_thresholds=quantized_double_punish,  # quantized_double_punish, quantized_mean_punish, create_instance_wise, static_timeout(5000)
        select_idx=create_top_k_sampling(3),
        scoring_fn=greedy_cross_acc,  # knapsack_rmse, greedy_rmse, knapsack_cross_acc, greedy_cross_acc
        thresh_breaking_conditions=[ThreshBreakingCondition('until_stab_0_9800', create_stability_breaking(0.98)), ThreshBreakingCondition('until_stab_0_9825', create_stability_breaking(0.9825)), ThreshBreakingCondition('until_stab_0_9850', create_stability_breaking(0.985)), ThreshBreakingCondition('until_stab_0_9875', create_stability_breaking(0.9875)), ThreshBreakingCondition('until_stab_0_9900', create_stability_breaking(0.99)), ThreshBreakingCondition('until_stab_0_9925', create_stability_breaking(0.9925)), ThreshBreakingCondition('until_stab_0_9950', create_stability_breaking(0.995)), ThreshBreakingCondition('until_stab_0_9975', create_stability_breaking(0.9975))],  # leave one if no breaking cond is needed (instance wise)
        temperatures=[],  # [0.5, 0.35, 0.25, 0.125, 0.09, 0.06125, 0.03075, 0.01530, 0.008, 0.004],
        rt_weights=[1],   # [1.0, 0.95, 1.1, 1.3, 1.5, 0.8, 1.6, 1.2, 1.4, 1.7, 1.05, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0],
        instance_selections=[choose_instances_random, variance_based_selection_1, variance_based_selection_2, highest_rt_selection, lowest_variance, highest_variance, lowest_variances_per_rt, lowest_rt_selection],  # choose_instances_random, variance_based_selection_1, variance_based_selection_2, highest_rt_selection, lowest_variance, highest_variance, lowest_variances_per_rt, lowest_rt_selection
        individual_solver_plots=False
    ),
    ExperimentConfig(
        filter_unsolvable=False,
        determine_thresholds=quantized_double_punish,  # quantized_double_punish, quantized_mean_punish, create_instance_wise, static_timeout(5000)
        select_idx=create_top_k_sampling(3),
        scoring_fn=greedy_cross_acc,  # knapsack_rmse, greedy_rmse, knapsack_cross_acc, greedy_cross_acc
        thresh_breaking_conditions=[ThreshBreakingCondition('until_stab_0_9800', create_stability_breaking(0.98)), ThreshBreakingCondition('until_stab_0_9825', create_stability_breaking(0.9825)), ThreshBreakingCondition('until_stab_0_9850', create_stability_breaking(0.985)), ThreshBreakingCondition('until_stab_0_9875', create_stability_breaking(0.9875)), ThreshBreakingCondition('until_stab_0_9900', create_stability_breaking(0.99)), ThreshBreakingCondition('until_stab_0_9925', create_stability_breaking(0.9925)), ThreshBreakingCondition('until_stab_0_9950', create_stability_breaking(0.995)), ThreshBreakingCondition('until_stab_0_9975', create_stability_breaking(0.9975))],  # leave one if no breaking cond is needed (instance wise)
        temperatures=[],  # [0.5, 0.35, 0.25, 0.125, 0.09, 0.06125, 0.03075, 0.01530, 0.008, 0.004],
        rt_weights=[1],   # [1.0, 0.95, 1.1, 1.3, 1.5, 0.8, 1.6, 1.2, 1.4, 1.7, 1.05, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0],
        instance_selections=[choose_instances_random, variance_based_selection_1, variance_based_selection_2, highest_rt_selection, lowest_variance, highest_variance, lowest_variances_per_rt, lowest_rt_selection],  # choose_instances_random, variance_based_selection_1, variance_based_selection_2, highest_rt_selection, lowest_variance, highest_variance, lowest_variances_per_rt, lowest_rt_selection
        individual_solver_plots=False
    ),
    ExperimentConfig(
        filter_unsolvable=True,
        determine_thresholds=quantized_double_punish,  # quantized_double_punish, quantized_mean_punish, create_instance_wise, static_timeout(5000)
        select_idx=create_top_k_sampling(9),
        scoring_fn=greedy_rmse,  # knapsack_rmse, greedy_rmse, knapsack_cross_acc, greedy_cross_acc
        thresh_breaking_conditions=[ThreshBreakingCondition('until_stab_0_9800', create_stability_breaking(0.98)), ThreshBreakingCondition('until_stab_0_9825', create_stability_breaking(0.9825)), ThreshBreakingCondition('until_stab_0_9850', create_stability_breaking(0.985)), ThreshBreakingCondition('until_stab_0_9875', create_stability_breaking(0.9875)), ThreshBreakingCondition('until_stab_0_9900', create_stability_breaking(0.99)), ThreshBreakingCondition('until_stab_0_9925', create_stability_breaking(0.9925)), ThreshBreakingCondition('until_stab_0_9950', create_stability_breaking(0.995)), ThreshBreakingCondition('until_stab_0_9975', create_stability_breaking(0.9975))],  # leave one if no breaking cond is needed (instance wise)
        temperatures=[],  # [0.5, 0.35, 0.25, 0.125, 0.09, 0.06125, 0.03075, 0.01530, 0.008, 0.004],
        rt_weights=[1],   # [1.0, 0.95, 1.1, 1.3, 1.5, 0.8, 1.6, 1.2, 1.4, 1.7, 1.05, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0],
        instance_selections=[choose_instances_random, variance_based_selection_1, variance_based_selection_2, highest_rt_selection, lowest_variance, highest_variance, lowest_variances_per_rt, lowest_rt_selection],  # choose_instances_random, variance_based_selection_1, variance_based_selection_2, highest_rt_selection, lowest_variance, highest_variance, lowest_variances_per_rt, lowest_rt_selection
        individual_solver_plots=False
    ),
    ExperimentConfig(
        filter_unsolvable=False,
        determine_thresholds=quantized_double_punish,  # quantized_double_punish, quantized_mean_punish, create_instance_wise, static_timeout(5000)
        select_idx=create_top_k_sampling(9),
        scoring_fn=greedy_rmse,  # knapsack_rmse, greedy_rmse, knapsack_cross_acc, greedy_cross_acc
        thresh_breaking_conditions=[ThreshBreakingCondition('until_stab_0_9800', create_stability_breaking(0.98)), ThreshBreakingCondition('until_stab_0_9825', create_stability_breaking(0.9825)), ThreshBreakingCondition('until_stab_0_9850', create_stability_breaking(0.985)), ThreshBreakingCondition('until_stab_0_9875', create_stability_breaking(0.9875)), ThreshBreakingCondition('until_stab_0_9900', create_stability_breaking(0.99)), ThreshBreakingCondition('until_stab_0_9925', create_stability_breaking(0.9925)), ThreshBreakingCondition('until_stab_0_9950', create_stability_breaking(0.995)), ThreshBreakingCondition('until_stab_0_9975', create_stability_breaking(0.9975))],  # leave one if no breaking cond is needed (instance wise)
        temperatures=[],  # [0.5, 0.35, 0.25, 0.125, 0.09, 0.06125, 0.03075, 0.01530, 0.008, 0.004],
        rt_weights=[1],   # [1.0, 0.95, 1.1, 1.3, 1.5, 0.8, 1.6, 1.2, 1.4, 1.7, 1.05, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0],
        instance_selections=[choose_instances_random, variance_based_selection_1, variance_based_selection_2, highest_rt_selection, lowest_variance, highest_variance, lowest_variances_per_rt, lowest_rt_selection],  # choose_instances_random, variance_based_selection_1, variance_based_selection_2, highest_rt_selection, lowest_variance, highest_variance, lowest_variances_per_rt, lowest_rt_selection
        individual_solver_plots=False
    ),
    ExperimentConfig(
        filter_unsolvable=True,
        determine_thresholds=quantized_double_punish,  # quantized_double_punish, quantized_mean_punish, create_instance_wise, static_timeout(5000)
        select_idx=create_top_k_sampling(9),
        scoring_fn=greedy_cross_acc,  # knapsack_rmse, greedy_rmse, knapsack_cross_acc, greedy_cross_acc
        thresh_breaking_conditions=[ThreshBreakingCondition('until_stab_0_9800', create_stability_breaking(0.98)), ThreshBreakingCondition('until_stab_0_9825', create_stability_breaking(0.9825)), ThreshBreakingCondition('until_stab_0_9850', create_stability_breaking(0.985)), ThreshBreakingCondition('until_stab_0_9875', create_stability_breaking(0.9875)), ThreshBreakingCondition('until_stab_0_9900', create_stability_breaking(0.99)), ThreshBreakingCondition('until_stab_0_9925', create_stability_breaking(0.9925)), ThreshBreakingCondition('until_stab_0_9950', create_stability_breaking(0.995)), ThreshBreakingCondition('until_stab_0_9975', create_stability_breaking(0.9975))],  # leave one if no breaking cond is needed (instance wise)
        temperatures=[],  # [0.5, 0.35, 0.25, 0.125, 0.09, 0.06125, 0.03075, 0.01530, 0.008, 0.004],
        rt_weights=[1],   # [1.0, 0.95, 1.1, 1.3, 1.5, 0.8, 1.6, 1.2, 1.4, 1.7, 1.05, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0],
        instance_selections=[choose_instances_random, variance_based_selection_1, variance_based_selection_2, highest_rt_selection, lowest_variance, highest_variance, lowest_variances_per_rt, lowest_rt_selection],  # choose_instances_random, variance_based_selection_1, variance_based_selection_2, highest_rt_selection, lowest_variance, highest_variance, lowest_variances_per_rt, lowest_rt_selection
        individual_solver_plots=False
    ),
    ExperimentConfig(
        filter_unsolvable=False,
        determine_thresholds=quantized_double_punish,  # quantized_double_punish, quantized_mean_punish, create_instance_wise, static_timeout(5000)
        select_idx=create_top_k_sampling(9),
        scoring_fn=greedy_cross_acc,  # knapsack_rmse, greedy_rmse, knapsack_cross_acc, greedy_cross_acc
        thresh_breaking_conditions=[ThreshBreakingCondition('until_stab_0_9800', create_stability_breaking(0.98)), ThreshBreakingCondition('until_stab_0_9825', create_stability_breaking(0.9825)), ThreshBreakingCondition('until_stab_0_9850', create_stability_breaking(0.985)), ThreshBreakingCondition('until_stab_0_9875', create_stability_breaking(0.9875)), ThreshBreakingCondition('until_stab_0_9900', create_stability_breaking(0.99)), ThreshBreakingCondition('until_stab_0_9925', create_stability_breaking(0.9925)), ThreshBreakingCondition('until_stab_0_9950', create_stability_breaking(0.995)), ThreshBreakingCondition('until_stab_0_9975', create_stability_breaking(0.9975))],  # leave one if no breaking cond is needed (instance wise)
        temperatures=[],  # [0.5, 0.35, 0.25, 0.125, 0.09, 0.06125, 0.03075, 0.01530, 0.008, 0.004],
        rt_weights=[1],   # [1.0, 0.95, 1.1, 1.3, 1.5, 0.8, 1.6, 1.2, 1.4, 1.7, 1.05, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0],
        instance_selections=[choose_instances_random, variance_based_selection_1, variance_based_selection_2, highest_rt_selection, lowest_variance, highest_variance, lowest_variances_per_rt, lowest_rt_selection],  # choose_instances_random, variance_based_selection_1, variance_based_selection_2, highest_rt_selection, lowest_variance, highest_variance, lowest_variances_per_rt, lowest_rt_selection
        individual_solver_plots=False
    ),
    ExperimentConfig(
        filter_unsolvable=True,
        determine_thresholds=quantized_double_punish,  # quantized_double_punish, quantized_mean_punish, create_instance_wise, static_timeout(5000)
        select_idx=create_top_k_sampling(27),
        scoring_fn=greedy_rmse,  # knapsack_rmse, greedy_rmse, knapsack_cross_acc, greedy_cross_acc
        thresh_breaking_conditions=[ThreshBreakingCondition('until_stab_0_9800', create_stability_breaking(0.98)), ThreshBreakingCondition('until_stab_0_9825', create_stability_breaking(0.9825)), ThreshBreakingCondition('until_stab_0_9850', create_stability_breaking(0.985)), ThreshBreakingCondition('until_stab_0_9875', create_stability_breaking(0.9875)), ThreshBreakingCondition('until_stab_0_9900', create_stability_breaking(0.99)), ThreshBreakingCondition('until_stab_0_9925', create_stability_breaking(0.9925)), ThreshBreakingCondition('until_stab_0_9950', create_stability_breaking(0.995)), ThreshBreakingCondition('until_stab_0_9975', create_stability_breaking(0.9975))],  # leave one if no breaking cond is needed (instance wise)
        temperatures=[],  # [0.5, 0.35, 0.25, 0.125, 0.09, 0.06125, 0.03075, 0.01530, 0.008, 0.004],
        rt_weights=[1],   # [1.0, 0.95, 1.1, 1.3, 1.5, 0.8, 1.6, 1.2, 1.4, 1.7, 1.05, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0],
        instance_selections=[choose_instances_random, variance_based_selection_1, variance_based_selection_2, highest_rt_selection, lowest_variance, highest_variance, lowest_variances_per_rt, lowest_rt_selection],  # choose_instances_random, variance_based_selection_1, variance_based_selection_2, highest_rt_selection, lowest_variance, highest_variance, lowest_variances_per_rt, lowest_rt_selection
        individual_solver_plots=False
    ),
    ExperimentConfig(
        filter_unsolvable=False,
        determine_thresholds=quantized_double_punish,  # quantized_double_punish, quantized_mean_punish, create_instance_wise, static_timeout(5000)
        select_idx=create_top_k_sampling(27),
        scoring_fn=greedy_rmse,  # knapsack_rmse, greedy_rmse, knapsack_cross_acc, greedy_cross_acc
        thresh_breaking_conditions=[ThreshBreakingCondition('until_stab_0_9800', create_stability_breaking(0.98)), ThreshBreakingCondition('until_stab_0_9825', create_stability_breaking(0.9825)), ThreshBreakingCondition('until_stab_0_9850', create_stability_breaking(0.985)), ThreshBreakingCondition('until_stab_0_9875', create_stability_breaking(0.9875)), ThreshBreakingCondition('until_stab_0_9900', create_stability_breaking(0.99)), ThreshBreakingCondition('until_stab_0_9925', create_stability_breaking(0.9925)), ThreshBreakingCondition('until_stab_0_9950', create_stability_breaking(0.995)), ThreshBreakingCondition('until_stab_0_9975', create_stability_breaking(0.9975))],  # leave one if no breaking cond is needed (instance wise)
        temperatures=[],  # [0.5, 0.35, 0.25, 0.125, 0.09, 0.06125, 0.03075, 0.01530, 0.008, 0.004],
        rt_weights=[1],   # [1.0, 0.95, 1.1, 1.3, 1.5, 0.8, 1.6, 1.2, 1.4, 1.7, 1.05, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0],
        instance_selections=[choose_instances_random, variance_based_selection_1, variance_based_selection_2, highest_rt_selection, lowest_variance, highest_variance, lowest_variances_per_rt, lowest_rt_selection],  # choose_instances_random, variance_based_selection_1, variance_based_selection_2, highest_rt_selection, lowest_variance, highest_variance, lowest_variances_per_rt, lowest_rt_selection
        individual_solver_plots=False
    ),
    ExperimentConfig(
        filter_unsolvable=True,
        determine_thresholds=quantized_double_punish,  # quantized_double_punish, quantized_mean_punish, create_instance_wise, static_timeout(5000)
        select_idx=create_top_k_sampling(27),
        scoring_fn=greedy_cross_acc,  # knapsack_rmse, greedy_rmse, knapsack_cross_acc, greedy_cross_acc
        thresh_breaking_conditions=[ThreshBreakingCondition('until_stab_0_9800', create_stability_breaking(0.98)), ThreshBreakingCondition('until_stab_0_9825', create_stability_breaking(0.9825)), ThreshBreakingCondition('until_stab_0_9850', create_stability_breaking(0.985)), ThreshBreakingCondition('until_stab_0_9875', create_stability_breaking(0.9875)), ThreshBreakingCondition('until_stab_0_9900', create_stability_breaking(0.99)), ThreshBreakingCondition('until_stab_0_9925', create_stability_breaking(0.9925)), ThreshBreakingCondition('until_stab_0_9950', create_stability_breaking(0.995)), ThreshBreakingCondition('until_stab_0_9975', create_stability_breaking(0.9975))],  # leave one if no breaking cond is needed (instance wise)
        temperatures=[],  # [0.5, 0.35, 0.25, 0.125, 0.09, 0.06125, 0.03075, 0.01530, 0.008, 0.004],
        rt_weights=[1],   # [1.0, 0.95, 1.1, 1.3, 1.5, 0.8, 1.6, 1.2, 1.4, 1.7, 1.05, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0],
        instance_selections=[choose_instances_random, variance_based_selection_1, variance_based_selection_2, highest_rt_selection, lowest_variance, highest_variance, lowest_variances_per_rt, lowest_rt_selection],  # choose_instances_random, variance_based_selection_1, variance_based_selection_2, highest_rt_selection, lowest_variance, highest_variance, lowest_variances_per_rt, lowest_rt_selection
        individual_solver_plots=False
    ),
    ExperimentConfig(
        filter_unsolvable=False,
        determine_thresholds=quantized_double_punish,  # quantized_double_punish, quantized_mean_punish, create_instance_wise, static_timeout(5000)
        select_idx=create_top_k_sampling(27),
        scoring_fn=greedy_cross_acc,  # knapsack_rmse, greedy_rmse, knapsack_cross_acc, greedy_cross_acc
        thresh_breaking_conditions=[ThreshBreakingCondition('until_stab_0_9800', create_stability_breaking(0.98)), ThreshBreakingCondition('until_stab_0_9825', create_stability_breaking(0.9825)), ThreshBreakingCondition('until_stab_0_9850', create_stability_breaking(0.985)), ThreshBreakingCondition('until_stab_0_9875', create_stability_breaking(0.9875)), ThreshBreakingCondition('until_stab_0_9900', create_stability_breaking(0.99)), ThreshBreakingCondition('until_stab_0_9925', create_stability_breaking(0.9925)), ThreshBreakingCondition('until_stab_0_9950', create_stability_breaking(0.995)), ThreshBreakingCondition('until_stab_0_9975', create_stability_breaking(0.9975))],  # leave one if no breaking cond is needed (instance wise)
        temperatures=[],  # [0.5, 0.35, 0.25, 0.125, 0.09, 0.06125, 0.03075, 0.01530, 0.008, 0.004],
        rt_weights=[1],   # [1.0, 0.95, 1.1, 1.3, 1.5, 0.8, 1.6, 1.2, 1.4, 1.7, 1.05, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0],
        instance_selections=[choose_instances_random, variance_based_selection_1, variance_based_selection_2, highest_rt_selection, lowest_variance, highest_variance, lowest_variances_per_rt, lowest_rt_selection],  # choose_instances_random, variance_based_selection_1, variance_based_selection_2, highest_rt_selection, lowest_variance, highest_variance, lowest_variances_per_rt, lowest_rt_selection
        individual_solver_plots=False
    )
]