import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


class PlotGenerator:
    results = []

    def __init__(self, git_hash, exp_config, experiment=None):
        self.git_hash = git_hash
        self.exp_config = exp_config
        if experiment is None:
            self.out_dir = f"./plots/{self.git_hash}"
        else:
            self.out_dir = f"./plots/{self.git_hash}/{experiment}"
        self.experiment = experiment

    def plot_avg_results(self, df: pd.DataFrame, num_samples):
        # filter
        wanted_measurements = ["determine_timeouts_true_acc_v2", "determine_timeouts_cross_acc"]
        for sel_fn in self.exp_config.instance_selections:
            wanted_measurements.append(f"{sel_fn.__name__}_true_acc_v2")
        # Now plot:

        # 1) define grid
        x_min, x_max = df['runtime_fraction'].min(), df['runtime_fraction'].max()
        x_grid = np.linspace(x_min, x_max, num_samples)

        # 2–3) interpolate each (solver, measurement)
        recs = []
        for (solver, meas), grp in df.groupby(['solver', 'measurement']):
            # sort just in case
            xs = grp['runtime_fraction'].values
            ys = grp['value'].values
            sort_idx = np.argsort(xs)
            xs, ys = xs[sort_idx], ys[sort_idx]
            # linear interp, out-of-bounds → NaN
            y_grid = np.interp(
                x_grid,
                xs,
                ys,
                left=np.nan,
                right=np.nan
            )
            for x_val, y_val in zip(x_grid, y_grid):
                recs.append({
                    'solver': solver,
                    'measurement': meas,
                    'runtime_fraction': x_val,
                    'value': y_val
                })

        interp_df = pd.DataFrame.from_records(recs)

        # 4) plot
        plt.figure(figsize=(8, 6))
        sns.lineplot(
            data=interp_df,
            x='runtime_fraction',
            y='value',
            hue='measurement',
            hue_order=wanted_measurements,
            estimator='mean',
            errorbar=('pi', 90),     # 90 % prediction interval (5th–95th pct)
            legend='full'
        )

        out_path = os.path.join(self.out_dir, "average_results.png")

        # create the directory (and any parents) if it doesn't exist
        os.makedirs(self.out_dir, exist_ok=True)

        plt.savefig(out_path, dpi=300, bbox_inches="tight")

        # Clean up
        plt.close()

    def create_average_plot(self, df: pd.DataFrame, wanted_measurements, legend_label, num_samples=500):
        # 0) filtering
        df = df[df['measurement'].isin(wanted_measurements)]

        # 1) define grid
        x_min, x_max = df['runtime_fraction'].min(), df['runtime_fraction'].max()
        x_grid = np.linspace(x_min, x_max, num_samples)

        # 2–3) interpolate each (solver, measurement)
        recs = []
        for (solver, meas), grp in df.groupby(['solver', 'measurement']):
            # sort just in case
            xs = grp['runtime_fraction'].values
            ys = grp['value'].values
            sort_idx = np.argsort(xs)
            xs, ys = xs[sort_idx], ys[sort_idx]
            # linear interp, out-of-bounds → NaN
            y_grid = np.interp(
                x_grid,
                xs,
                ys,
                left=np.nan,
                right=np.nan
            )
            for x_val, y_val in zip(x_grid, y_grid):
                recs.append({
                    'solver': solver,
                    'measurement': meas,
                    'runtime_fraction': x_val,
                    'value': y_val
                })

        interp_df = pd.DataFrame.from_records(recs)

        # 4) plot
        sns.lineplot(
            data=interp_df,
            x='runtime_fraction',
            y='value',
            estimator='mean',
            errorbar=('pi', 90),     # 90 % prediction interval (5th–95th pct)
            legend=False,
            label=legend_label
        )

    def plot_solver_results(
        self, all_results, solver_string
    ):
        # construct df
        df = pd.DataFrame.from_records(all_results)

        # filtering
        wanted_solver = [solver_string]
        df_sub = df[df["solver"].isin(wanted_solver)]
        wanted_measurements = ["determine_timeouts_true_acc_v2", "determine_timeouts_cross_acc"]
        for sel_fn in self.exp_config.instance_selections:
            wanted_measurements.append(f"{sel_fn.__name__}_true_acc_v2")
        # Now plot:
        plt.figure(figsize=(8, 6))
        sns.lineplot(
            data=df_sub,
            x="runtime_fraction",
            y="value",
            hue="measurement",
            hue_order=wanted_measurements,
        )
        plt.title(f"dynamic timeouts for solver {solver_string}")
        plt.legend(title="Measurement")

        out_path = os.path.join(self.out_dir, f"{solver_string}_results.png")

        # create the directory (and any parents) if it doesn't exist
        os.makedirs(self.out_dir, exist_ok=True)

        plt.savefig(out_path, dpi=300, bbox_inches="tight")

        # Clean up
        plt.close()
        """ # create main plot and a twin y-axis
        fig, ax1 = plt.subplots()

        ax2 = ax1.twinx()

        # grab the default Tab10 palette (10 colors)
        tab10 = plt.cm.get_cmap("tab10").colors

        # remove first 2 colors form ax2 (blue and green)
        ax2.set_prop_cycle("color", tab10[2:])

        if len(solver_results) > 0:
            ax1.plot(
                solver_results["runtime_frac"], solver_results["diff"],
                color='g',
                label="diff"
            )
            ax1.set_ylabel("diff", color='g')

            ax2.plot(
                solver_results["runtime_frac"], solver_results["cross_acc"],
                color='b',
                label="cross_acc_timeout_selection"
            )
            ax2.plot(
                solver_results["runtime_frac"], solver_results["true_acc"],
                color='r',
                label="true_acc_timeout_selection"
            )
        for function_name, results in instance_selection_results.items():
            if len(results) == 0:
                continue
            # last results dataframe is from the current solver
            solver_result = results[-1]
            ax2.plot(
                solver_result["runtime_frac"], solver_result["true_acc"],
                label=function_name
            )
        ax2.set_ylabel("cross_acc", color='b')
        ax2.set_ylabel("true_acc", color='r')

        plt.title(f"dynamic timeouts for solver {solver_string}")

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()

        # one legend on ax1 (or fig.legend())
        ax1.legend(h1 + h2, l1 + l2, loc='best')

        # optional: add grids and legends
        ax1.grid(True)
        ax1.set_xlabel("runtime fraction")
        fig.tight_layout() """

    def create_solver_plot(self, df: pd.DataFrame, wanted_measurements, legend_label, solver_name: str):
        """
        Plot the raw runtime_fraction-value curves for a given solver and the given measurements.
        No interpolation, no averaging across solvers.
        """
        # 0) filtering
        sub = df[
            (df['solver'] == solver_name) &
            (df['measurement'].isin(wanted_measurements))
        ].copy()

        if sub.empty:
            print(f"No data found for solver {solver_name} with measurements {wanted_measurements}")
            return

        # 1) sort values for nicer lines
        sub = sub.sort_values(by=['measurement', 'runtime_fraction'])

        # 2) plot raw curves
        sns.lineplot(
            data=sub,
            x='runtime_fraction',
            y='value',
            hue='measurement',   # separate lines for true_acc / cross_acc
            marker="o",          # emphasize actual sample points
            legend=True
        )

    def get_all_measurements(self, dfs):
        result_string = ""

        deltas = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        filterings = [True, True, True, True, True, True, False, False, False, False, False, False]

        deltas.reverse()
        filterings.reverse()

        for df in dfs:
            delta = deltas.pop()
            filtering = filterings.pop()
            for selection_method in ['choose_instances_random', 'variance_based_selection_1', 'variance_based_selection_2', 'highest_rt_selection', 'lowest_variance', 'highest_variance', 'lowest_variances_per_rt', 'lowest_rt_selection']:
                result_string += self.print_lowest_rf_cross_acc(df, selection_method, 0.9, f'filter={filtering}; δ={delta}; sel={selection_method}; breaking={0.9}')
                result_string += self.print_lowest_rf_cross_acc(df, selection_method, 0.925, f'filter={filtering}; δ={delta}; sel={selection_method}; breaking={0.925}')
                result_string += self.print_lowest_rf_cross_acc(df, selection_method, 0.95, f'filter={filtering}; δ={delta}; sel={selection_method}; breaking={0.95}')
                result_string += self.print_lowest_rf_cross_acc(df, selection_method, 0.975, f'filter={filtering}; δ={delta}; sel={selection_method}; breaking={0.975}')

        print()
        print("combined:")
        print(result_string)

        results_df = pd.DataFrame(self.results, columns=["x", "y", "std_x", "std_y", "label"])

        pareto = self.pareto_front(results_df)

        self.plot_df(pareto)

        self.results = []

    def plot_df(self, df):
        sns.set_theme(style="whitegrid")

        # Keep track of handles and labels for legend
        handles_labels = []

        # Plot each label group
        for lbl, df_group in df.groupby("label", sort=False):  # sort=False preserves order in df
            h = plt.errorbar(
                df_group["x"], df_group["y"],
                xerr=df_group["std_x"], yerr=df_group["std_y"],
                fmt="o", capsize=4, label=lbl
            )
            handles_labels.append((h[0], lbl))  # store handle and label

        # Extract unique labels in the order they appear in the DataFrame
        unique_labels_ordered = []
        seen = set()
        for lbl in df["label"]:
            if lbl not in seen:
                unique_labels_ordered.append(lbl)
                seen.add(lbl)

        # Build legend using ordered handles
        ordered_handles = [h for h, lbl in handles_labels if lbl in unique_labels_ordered]
        plt.legend(ordered_handles, unique_labels_ordered, title="Label")

    def print_lowest_rf_cross_acc(self, df: pd.DataFrame, wanted_measurement, threshold: float, label) -> pd.DataFrame:
        """
        For each solver, find the smallet runtime_fraction where cross_acc > threshold.
        Also print the corresponding true_acc at that runtime_fraction.
        Additionally, compute and print averages and std deviations across solvers.
        """
        # Keep only cross_acc rows
        cross = df.loc[(df["measurement"] == f"{wanted_measurement}_cross_acc") & df["value"].notna(),
                       ["solver", "runtime_fraction", "value"]] \
                  .rename(columns={"value": "cross_acc"})

        true_v1 = df.loc[(df["measurement"] == f"{wanted_measurement}_true_acc_v1") & df["value"].notna(),
                      ["solver", "runtime_fraction", "value"]] \
                 .rename(columns={"value": "true_acc_v1"})

        true_v2 = df.loc[(df["measurement"] == f"{wanted_measurement}_true_acc_v2") & df["value"].notna(),
                      ["solver", "runtime_fraction", "value"]] \
                 .rename(columns={"value": "true_acc_v2"})

        solvers = df['solver'].unique()
        records = []

        for solver in solvers:
            cross_s = cross[cross['solver'] == solver]
            true_s_v1 = true_v1[true_v1['solver'] == solver]
            true_s_v2 = true_v2[true_v2['solver'] == solver]

            if cross_s.empty or true_s_v1.empty or true_s_v2.empty:
                continue  # skip solver if no data at all

            # check which points cross the threshold
            above_thresh = cross_s[cross_s["cross_acc"] > threshold]

            if not above_thresh.empty:
                # pick smallest runtime_fraction
                min_rf = above_thresh["runtime_fraction"].min()
            else:
                # pick last sampled point (largest runtime_fraction)
                min_rf = cross_s["runtime_fraction"].max()

            # get corresponding values
            cross_val = cross_s.loc[cross_s["runtime_fraction"] == min_rf, "cross_acc"].values[0]
            true_val_v1 = true_s_v1.loc[true_s_v1["runtime_fraction"] == min_rf, "true_acc_v1"].values[0]
            true_val_v2 = true_s_v2.loc[true_s_v2["runtime_fraction"] == min_rf, "true_acc_v2"].values[0]

            records.append({
                "solver": solver,
                "min_runtime_fraction": min_rf,
                "cross_acc": cross_val,
                "true_acc_v1": true_val_v1,
                "true_acc_v2": true_val_v2
            })

            print(f"{solver}: runtime_fraction = {min_rf:.4f}, cross_acc = {cross_val:.4f}, true_acc_v1 = {true_val_v1:.4f},  true_acc_v2 = {true_val_v2:.4f}")

        out = pd.DataFrame.from_records(records)

        # ---- averages and stds ----
        avg_rf = out["min_runtime_fraction"].mean()
        std_rf = out["min_runtime_fraction"].std(ddof=1)
        avg_true_v1 = out["true_acc_v1"].mean()
        std_true_v1 = out["true_acc_v1"].std(ddof=1)
        avg_true_v2 = out["true_acc_v2"].mean()
        std_true_v2 = out["true_acc_v2"].std(ddof=1)

        print("\n=== Summary over solvers ===")
        print(f"Runtime_fraction: mean = {avg_rf:.3f}, std = {std_rf:.3f}")
        print(f"True_acc_v1     : mean = {avg_true_v1:.3f}, std = {std_true_v1:.3f}")
        print(f"True_acc_v2     : mean = {avg_true_v2:.3f}, std = {std_true_v2:.3f}")

        # (x, y, std_x, std_y)
        self.results.append((avg_rf, avg_true_v1, std_rf, std_true_v1, label + '; v1'))
        self.results.append((avg_rf, avg_true_v2, std_rf, std_true_v2, label + '; v2'))

        return f"$0.4$ & {wanted_measurement} & ${threshold}$ & ${avg_rf:.3f} \pm {std_rf:.3f}$ & ${avg_true_v1:.3f} \pm {std_true_v1:.3f}$ & ${avg_true_v2:.3f} \pm {std_true_v2:.3f}$ \\\\\n\n"

    def pareto_front(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return the Pareto front from a DataFrame with columns ["x", "y", "std_x", "std_y"].
        Optimization: smaller x is better, larger y is better.
        """
        pareto_points = []
        for i, row in df.iterrows():
            x_i, y_i = row["x"], row["y"]

            # A point is on the Pareto front if no other point dominates it
            dominated = False
            for j, other in df.iterrows():
                if j == i:
                    continue
                x_j, y_j = other["x"], other["y"]

                if (x_j <= x_i and y_j >= y_i) and (x_j < x_i or y_j > y_i):
                    dominated = True
                    break

            if not dominated:
                pareto_points.append(i)

        pareto_df = df.loc[pareto_points].reset_index(drop=True)
        sorted_df = pareto_df.sort_values(by="x").reset_index(drop=True)

        return sorted_df

    def create_progress_plot(self):
        linear_only_diff = pd.read_pickle("./pickle/061637b4_df.pkl.gz", compression='gzip')
        linear_knapsack = pd.read_pickle("./pickle/39e39172_df.pkl.gz", compression='gzip')

        # sim weight
        sim_0_8 = pd.read_pickle("./pickle/106119e9_rt_weigth_0_8_temp_None.pkl.gz", compression='gzip')
        sim_0_95 = pd.read_pickle("./pickle/106119e9_rt_weigth_0_95_temp_None.pkl.gz", compression='gzip')
        sim_1_0 = pd.read_pickle("./pickle/106119e9_rt_weigth_1_0_temp_None.pkl.gz", compression='gzip')
        sim_1_1 = pd.read_pickle("./pickle/106119e9_rt_weigth_1_1_temp_None.pkl.gz", compression='gzip')
        sim_1_2 = pd.read_pickle("./pickle/106119e9_rt_weigth_1_2_temp_None.pkl.gz", compression='gzip')
        sim_1_3 = pd.read_pickle("./pickle/106119e9_rt_weigth_1_3_temp_None.pkl.gz", compression='gzip')
        sim_1_4 = pd.read_pickle("./pickle/106119e9_rt_weigth_1_4_temp_None.pkl.gz", compression='gzip')
        sim_1_5 = pd.read_pickle("./pickle/106119e9_rt_weigth_1_5_temp_None.pkl.gz", compression='gzip')
        sim_2_0 = pd.read_pickle("./pickle/106119e9_rt_weigth_2_0_temp_None.pkl.gz", compression='gzip')
        sim_2_5 = pd.read_pickle("./pickle/106119e9_rt_weigth_2_5_temp_None.pkl.gz", compression='gzip')
        sim_3_0 = pd.read_pickle("./pickle/106119e9_rt_weigth_3_0_temp_None.pkl.gz", compression='gzip')

        # whole instace (static 5000s timeout)
        whole = pd.read_pickle("./pickle/d03ff829_rt_weigth_1_temp_None.pkl.gz", compression='gzip')

        
        knapsack_break_after_0_96 = pd.read_pickle("./pickle/0a083288_rt_weigth_1_temp_None.pkl.gz", compression='gzip')
        knapsack_break_after_0_97 = pd.read_pickle("./pickle/341dfcaa_rt_weigth_1_temp_None.pkl.gz", compression='gzip')
        knapsack_break_after_0_98 = pd.read_pickle("./pickle/939f33b9_rt_weigth_1_temp_None.pkl.gz", compression='gzip')
        knapsack_break_after_0_99 = pd.read_pickle("./pickle/b5ee1717_rt_weigth_1_temp_None.pkl.gz", compression='gzip')
        knapsack_break_after_1_00 = pd.read_pickle("./pickle/96285047_rt_weigth_1_temp_None.pkl.gz", compression='gzip')

        knapsack_dont_break = pd.read_pickle("./pickle/2eaaeefa_rt_weigth_1_temp_None.pkl.gz", compression='gzip')
        min_rmse_dont_break = pd.read_pickle("./pickle/4d013e7e_rt_weigth_1_temp_None.pkl.gz", compression='gzip')
        min_cross_acc_dont_break = pd.read_pickle("./pickle/ec8f33d8_rt_weigth_1_temp_None.pkl.gz", compression='gzip')

        delta_0_4 = pd.read_pickle("./pickle/e92d3806_rt_weigth_1_temp_None.pkl.gz", compression='gzip')
        delta_0_5 = pd.read_pickle("./pickle/3554615b_rt_weigth_1_temp_None.pkl.gz", compression='gzip')
        delta_0_6 = pd.read_pickle("./pickle/a00e8733_rt_weigth_1_temp_None.pkl.gz", compression='gzip')
        delta_0_7 = pd.read_pickle("./pickle/1f39d6b3_rt_weigth_1_temp_None.pkl.gz", compression='gzip')
        delta_0_8 = pd.read_pickle("./pickle/c8a0ef69_rt_weigth_1_temp_None.pkl.gz", compression='gzip')
        delta_0_9 = pd.read_pickle("./pickle/c6197b53_rt_weigth_1_temp_None.pkl.gz", compression='gzip')        

        delta_0_4_no_filter = pd.read_pickle("./pickle/021eda4a_rt_weigth_1_temp_None.pkl.gz", compression='gzip')
        delta_0_5_no_filter = pd.read_pickle("./pickle/71d395b7_rt_weigth_1_temp_None.pkl.gz", compression='gzip')
        delta_0_6_no_filter = pd.read_pickle("./pickle/9810d8e2_rt_weigth_1_temp_None.pkl.gz", compression='gzip')
        delta_0_7_no_filter = pd.read_pickle("./pickle/d2a147f4_rt_weigth_1_temp_None.pkl.gz", compression='gzip')
        delta_0_8_no_filter = pd.read_pickle("./pickle/73ddf9ea_rt_weigth_1_temp_None.pkl.gz", compression='gzip')
        delta_0_9_no_filter = pd.read_pickle("./pickle/3e77ebd2_rt_weigth_1_temp_None.pkl.gz", compression='gzip')

        al_low_delta_rt = [0.0541, 0.1035]
        al_high_delta_acc = [0.9048, 0.9233]

        plt.figure(figsize=(10, 6))

        self.get_all_measurements([delta_0_4, delta_0_5, delta_0_6, delta_0_7, delta_0_8, delta_0_9, delta_0_4_no_filter, delta_0_5_no_filter, delta_0_6_no_filter, delta_0_7_no_filter, delta_0_8_no_filter, delta_0_9_no_filter])
        #self.create_solver_plot(delta_0_4, ['choose_instances_random_cross_acc', 'choose_instances_random_true_acc_v2'], "avg plot true acc", '0_CaDiCaL_DVDL_V1')
        #self.create_average_plot(delta_0_4, ['choose_instances_random_cross_acc'], "avg plot cross acc")

        #self.create_average_plot(min_rmse_dont_break, ['determine_timeouts_stability'], "stability")
        #self.create_average_plot(min_rmse_dont_break, ['determine_timeouts_true_acc'], "min diff determine timeout true acc")
        #self.create_average_plot(min_rmse_dont_break, ['determine_timeouts_cross_acc'], "min diff determine timeout cross acc")


        #self.create_average_plot(knapsack_dont_break, ['determine_timeouts_true_acc'], "knap determine timeout true acc")
        #self.create_average_plot(knapsack_dont_break, ['determine_timeouts_cross_acc'], "knap determine timeout cross acc")
        
        #self.create_average_plot(min_rmse_dont_break, ['determine_timeouts_diff'], "knap determine timeout diff")
        #self.create_average_plot(min_rmse_dont_break, ['determine_timeouts_stability'], "min rmse determine timeout stability")
        #self.create_average_plot(min_cross_acc_dont_break, ['determine_timeouts_rmse_stability'], "cross acc determine timeout stability")
        #self.create_average_plot(min_cross_acc_dont_break, ['choose_instances_random_true_acc'], "true acc")
        #self.create_average_plot(min_cross_acc_dont_break, ['choose_instances_random_cross_acc'], "cross acc")
        
        
        #self.create_average_plot(min_rmse_dont_break, ['choose_instances_random_true_acc'], "rmse true acc")
        #self.create_average_plot(min_rmse_dont_break, ['choose_instances_random_cross_acc'], "rmse cross acc")



        #self.create_average_plot(whole, ['choose_instances_random_true_acc'], "random choosing whole instances")
        #self.create_average_plot(knapsack_dont_break, ['choose_instances_random_true_acc'], "random choosing knap")
        #self.create_average_plot(knapsack_break_after_1_00, ['choose_instances_random_true_acc'], "break after 1.00")
        #self.create_average_plot(knapsack_break_after_0_99, ['choose_instances_random_true_acc'], "break after 0.99")
        #self.create_average_plot(knapsack_break_after_0_98, ['choose_instances_random_true_acc'], "break after 0.98")
        #self.create_average_plot(knapsack_break_after_0_97, ['choose_instances_random_true_acc'], "break after 0.97 true acc")
        #self.create_average_plot(knapsack_break_after_0_97, ['choose_instances_random_cross_acc'], "break after 0.97 cross acc")
        
        #self.create_average_plot(knapsack_break_after_0_96, ['choose_instances_random_true_acc'], "break after 0.96")


        #self.create_average_plot(linear_only_diff, ['choose_instances_random_true_acc'], "random choosing min diff")
        #self.create_average_plot(linear_knapsack, ['choose_instances_random_true_acc'], "random choosing knapsack")
        #self.create_average_plot(linear_only_diff, ['determine_timeouts_true_acc'], "precalc true acc only min diff")
        #self.create_average_plot(linear_knapsack, ['determine_timeouts_true_acc'], "precalc true acc knapsack")
        #self.create_average_plot(linear_knapsack, ['determine_timeouts_diff'], "RMSQ knapsack")
        #self.create_average_plot(linear_only_diff, ['determine_timeouts_diff'], "only min RMSQ")
#
        #self.create_average_plot(sim_0_8, ['determine_timeouts_diff'], "score weigth 0.8")
        #self.create_average_plot(sim_1_0, ['determine_timeouts_diff'], "score weigth 1.0")
        #self.create_average_plot(sim_1_5, ['determine_timeouts_diff'], "score weigth 1.5")
        #self.create_average_plot(sim_2_0, ['determine_timeouts_diff'], "score weigth 2.0")
        #self.create_average_plot(sim_2_5, ['determine_timeouts_diff'], "score weigth 2.5")
        #self.create_average_plot(sim_3_0, ['determine_timeouts_diff'], "score weigth 3.0")
        
        #self.create_average_plot(sim_0_8, ['choose_instances_random_true_acc'], "score weigth 0.8 true acc")
        #self.create_average_plot(sim_1_0, ['choose_instances_random_true_acc'], "score weigth 1.0 true acc")
        #self.create_average_plot(sim_1_5, ['choose_instances_random_true_acc'], "score weigth 1.5 true acc")
        #self.create_average_plot(sim_2_0, ['choose_instances_random_true_acc'], "score weigth 2.0 true acc")
        #self.create_average_plot(sim_2_5, ['choose_instances_random_true_acc'], "score weigth 2.5 true acc")
        #self.create_average_plot(sim_3_0, ['choose_instances_random_true_acc'], "score weigth 3.0 true acc")
        


        #self.create_average_plot(linear_only_diff, ['determine_timeouts_cross_acc'], "precalc cross acc")


        #whole instances
        #plt.plot(random_baseline_whole_instances_runtime_frac, random_baseline_whole_instances_true_acc, label="random instances")
        #plt.plot(random_baseline_whole_instances_runtime_frac_2, random_baseline_whole_instances_true_acc_2, label="random instances sampling")
        #plt.plot(sub_optimal_variance_based_selection_diff_runtime_fraction, sub_optimal_variance_based_selection_diff_true_acc, label="variance-based-selection")
        #plt.plot(h_5e935b6f_lowest_rt_selection_runtime_frac_rt_weight_1_temp_None, h_5e935b6f_lowest_rt_selection_true_acc_rt_weight_1_temp_None, label="select instance with smallest runtime")
        #plt.plot(h_5e935b6f_lowest_rt_selection_runtime_frac_rt_weight_1_temp_None, h_5e935b6f_lowest_rt_selection_cross_acc_rt_weight_1_temp_None, label="select smallest runtime_cross acc")


        
        #plt.plot(optimal_variance_based_selection_diff_runtime_fraction, optimal_variance_based_selection_diff_true_acc, label="optimal variance-based-selection")
        #plt.plot(random_baseline_dynamic_timeout_runtime_frac, random_baseline_dynamic_timeout_true_acc, label="random dynamic timeout")
        #plt.plot(variance_based_selection_runtime_frac, variance_based_selection_true_acc, label="variance based selection first run")
        #plt.plot(dynamic_timeout_optimized_runtime_frac, dynamic_timeout_optimized_true_acc, label="dynamic timeout optimized")
        #plt.plot(dynamic_timeout_quantized_selection_runtime_fraction, dynamic_timeout_quantized_selection_true_acc, label="min diff*rt")
        #plt.plot(dynamic_timeout_quantized_min_diff_runtime_fraction, dynamic_timeout_quantized_min_diff_true_acc, label="min diff")
        #plt.plot(quantized_diff_var_sel_runtime_fraction, quantized_diff_var_sel_true_acc, label="choose variance based after timeout precalculation")
        #plt.plot(quantized_diff_precalc_runtime_fraction, quantized_diff_precalc_true_acc, label="timeout precalculation true accuracy")


        #plt.plot(repl_quantized_diff_var_sel_runtime_fraction, repl_quantized_diff_var_sel_true_acc, label="repl choose variance based after timeout precalculation")
        #plt.plot(temp_0_01_runtime_fraction, temp_0_01_true_acc, label="with temp of 0.01")

        # compare rt_weights
        #plt.plot(h_2573bdc1_timeout_precalc_runtime_frac_inf, h_2573bdc1_timeout_precalc_true_acc_inf, label="rt_weight='inf'")
        #plt.plot(h_2573bdc1_timeout_precalc_runtime_frac_inf, h_2573bdc1_timeout_precalc_cross_acc_inf, label="rt_weight='inf' cross acc")
        #plt.plot(h_4965603a_timeout_precalc_runtime_frac_100, h_4965603a_timeout_precalc_true_acc_100, label="rt_weight=100")
        #plt.plot(h_4965603a_timeout_precalc_runtime_frac_10, h_4965603a_timeout_precalc_true_acc_10, label="rt_weight=10")
        #plt.plot(h_4965603a_timeout_precalc_runtime_frac_1, h_4965603a_timeout_precalc_true_acc_1, label="rt_weight=1")
        #plt.plot(h_4965603a_timeout_precalc_runtime_frac_0_1, h_4965603a_timeout_precalc_true_acc_0_1, label="rt_weight=0.1")
        #plt.plot(h_4965603a_timeout_precalc_runtime_frac_0_01, h_4965603a_timeout_precalc_true_acc_0_01, label="rt_weight=0.01")
        #plt.plot(h_4965603a_timeout_precalc_runtime_frac_0_01, h_4965603a_timeout_precalc_cross_acc_0_01, label="rt_weight=0.01 cross acc")
        #plt.plot(h_4965603a_timeout_precalc_runtime_frac_0_001, h_4965603a_timeout_precalc_true_acc_0_001, label="rt_weight=0.001")       
        #plt.plot(h_4965603a_timeout_precalc_runtime_frac_0_0001, h_4965603a_timeout_precalc_true_acc_0_0001, label="rt_weight=0.0001")      
        #plt.plot(h_4965603a_timeout_precalc_runtime_frac_1e_05, h_4965603a_timeout_precalc_true_acc_1e_05, label="rt_weight=1e05")
        #plt.plot(h_4965603a_timeout_precalc_runtime_frac_1e_06, h_4965603a_timeout_precalc_true_acc_1e_06, label="rt_weight=1e06")
        #plt.plot(dynamic_timeout_quantized_min_diff_runtime_fraction_2, dynamic_timeout_quantized_min_diff_true_acc_2, label="rt_weight=0")
        

        # compare temps rt_weight_inf.png
        #plt.plot(h_1db3f33f_timeout_precalc_runtime_frac_rt_weight_inf_temp_10, h_1db3f33f_timeout_precalc_true_acc_rt_weight_inf_temp_10, label="temp=10")
        #plt.plot(h_1db3f33f_timeout_precalc_runtime_frac_rt_weight_inf_temp_1, h_1db3f33f_timeout_precalc_true_acc_rt_weight_inf_temp_1, label="temp=1")
        #plt.plot(h_68b387a1_timeout_precalc_runtime_frac_rt_weight_inf_temp_0_5, h_68b387a1_timeout_precalc_true_acc_rt_weight_inf_temp_0_5, label="temp=0.5")
        #plt.plot(h_68b387a1_timeout_precalc_runtime_frac_rt_weight_inf_temp_0_35, h_68b387a1_timeout_precalc_true_acc_rt_weight_inf_temp_0_35, label="temp=0.35")
        #plt.plot(h_68b387a1_timeout_precalc_runtime_frac_rt_weight_inf_temp_0_25, h_68b387a1_timeout_precalc_true_acc_rt_weight_inf_temp_0_25, label="temp=0.25")
        #plt.plot(h_68b387a1_timeout_precalc_runtime_frac_rt_weight_inf_temp_0_125, h_68b387a1_timeout_precalc_true_acc_rt_weight_inf_temp_0_125, label="temp=0.125")
        #plt.plot(h_68b387a1_timeout_precalc_runtime_frac_rt_weight_inf_temp_0_09, h_68b387a1_timeout_precalc_true_acc_rt_weight_inf_temp_0_09, label="temp=0.09")
        #plt.plot(h_68b387a1_timeout_precalc_runtime_frac_rt_weight_inf_temp_0_06125, h_68b387a1_timeout_precalc_true_acc_rt_weight_inf_temp_0_06125, label="temp=0.06125")
        #plt.plot(h_68b387a1_timeout_precalc_runtime_frac_rt_weight_inf_temp_0_03075, h_68b387a1_timeout_precalc_true_acc_rt_weight_inf_temp_0_03075, label="temp=0.03075")
        ##plt.plot(h_68b387a1_timeout_precalc_runtime_frac_rt_weight_inf_temp_0_0153, h_68b387a1_timeout_precalc_true_acc_rt_weight_inf_temp_0_0153, label="temp=0.0153")
        ##plt.plot(h_68b387a1_timeout_precalc_runtime_frac_rt_weight_inf_temp_0_008, h_68b387a1_timeout_precalc_true_acc_rt_weight_inf_temp_0_008, label="temp=0.008")
        ##plt.plot(h_68b387a1_timeout_precalc_runtime_frac_rt_weight_inf_temp_0_004, h_68b387a1_timeout_precalc_true_acc_rt_weight_inf_temp_0_004, label="temp=0.004")
        ##plt.plot(h_1db3f33f_timeout_precalc_runtime_frac_rt_weight_inf_temp_1e_06, h_1db3f33f_timeout_precalc_true_acc_rt_weight_inf_temp_1e_06, label="temp=1e-06")
        #plt.plot(h_2573bdc1_timeout_precalc_runtime_frac_inf, h_2573bdc1_timeout_precalc_true_acc_inf, label="temp=0 (select best instance)")


        #plt.plot(h_1db3f33f_timeout_precalc_runtime_frac_rt_weight_inf_temp_10, h_1db3f33f_timeout_precalc_true_acc_rt_weight_inf_temp_10, color="r", label="temp=10")
        #plt.plot(h_1db3f33f_timeout_precalc_runtime_frac_rt_weight_inf_temp_10, h_1db3f33f_timeout_precalc_cross_acc_rt_weight_inf_temp_10, color="b", label="temp=10 cross acc")
        #plt.plot(h_1db3f33f_timeout_precalc_runtime_frac_rt_weight_inf_temp_1, h_1db3f33f_timeout_precalc_true_acc_rt_weight_inf_temp_1, color="r", label="temp=1")
        #plt.plot(h_1db3f33f_timeout_precalc_runtime_frac_rt_weight_inf_temp_1, h_1db3f33f_timeout_precalc_cross_acc_rt_weight_inf_temp_1, color="b", label="temp=1 cross acc")
        #plt.plot(h_68b387a1_timeout_precalc_runtime_frac_rt_weight_inf_temp_0_5, h_68b387a1_timeout_precalc_true_acc_rt_weight_inf_temp_0_5, color="r", label="temp=0.5")
        #plt.plot(h_68b387a1_timeout_precalc_runtime_frac_rt_weight_inf_temp_0_5, h_68b387a1_timeout_precalc_cross_acc_rt_weight_inf_temp_0_5, color="b", label="temp=0.5 cross acc")
        #
        #plt.plot(h_68b387a1_timeout_precalc_runtime_frac_rt_weight_inf_temp_0_35, h_68b387a1_timeout_precalc_true_acc_rt_weight_inf_temp_0_35, color="r", label="temp=0.35")
        #plt.plot(h_68b387a1_timeout_precalc_runtime_frac_rt_weight_inf_temp_0_35, h_68b387a1_timeout_precalc_cross_acc_rt_weight_inf_temp_0_35, color="b", label="temp=0.35 cross acc")
        #
        #plt.plot(h_68b387a1_timeout_precalc_runtime_frac_rt_weight_inf_temp_0_25, h_68b387a1_timeout_precalc_true_acc_rt_weight_inf_temp_0_25,  color="r", label="temp=0.25")
        #plt.plot(h_68b387a1_timeout_precalc_runtime_frac_rt_weight_inf_temp_0_25, h_68b387a1_timeout_precalc_cross_acc_rt_weight_inf_temp_0_25,  color="b", label="temp=0.25 cross")
        #
        #plt.plot(h_68b387a1_timeout_precalc_runtime_frac_rt_weight_inf_temp_0_125, h_68b387a1_timeout_precalc_true_acc_rt_weight_inf_temp_0_125,  color="r", label="temp=0.125")
        #plt.plot(h_68b387a1_timeout_precalc_runtime_frac_rt_weight_inf_temp_0_125, h_68b387a1_timeout_precalc_cross_acc_rt_weight_inf_temp_0_125,  color="b", label="temp=0.125 cross acc")
        #
        #plt.plot(h_68b387a1_timeout_precalc_runtime_frac_rt_weight_inf_temp_0_09, h_68b387a1_timeout_precalc_true_acc_rt_weight_inf_temp_0_09, color="r", label="temp=0.09")
        #plt.plot(h_68b387a1_timeout_precalc_runtime_frac_rt_weight_inf_temp_0_09, h_68b387a1_timeout_precalc_cross_acc_rt_weight_inf_temp_0_09, color="b", label="temp=0.09 cross acc")
        #
        #plt.plot(h_68b387a1_timeout_precalc_runtime_frac_rt_weight_inf_temp_0_06125, h_68b387a1_timeout_precalc_true_acc_rt_weight_inf_temp_0_06125, color="r", label="temp=0.06125")
        #plt.plot(h_68b387a1_timeout_precalc_runtime_frac_rt_weight_inf_temp_0_06125, h_68b387a1_timeout_precalc_cross_acc_rt_weight_inf_temp_0_06125, color="b", label="temp=0.06125 cross acc")
        #
        #plt.plot(h_68b387a1_timeout_precalc_runtime_frac_rt_weight_inf_temp_0_03075, h_68b387a1_timeout_precalc_true_acc_rt_weight_inf_temp_0_03075, color="r", label="temp=0.03075")
        #plt.plot(h_68b387a1_timeout_precalc_runtime_frac_rt_weight_inf_temp_0_03075, h_68b387a1_timeout_precalc_cross_acc_rt_weight_inf_temp_0_03075, color="b", label="temp=0.03075 cross")
        
        #plt.plot(h_68b387a1_timeout_precalc_runtime_frac_rt_weight_inf_temp_0_0153, h_68b387a1_timeout_precalc_true_acc_rt_weight_inf_temp_0_0153, label="temp=0.0153")
        #plt.plot(h_68b387a1_timeout_precalc_runtime_frac_rt_weight_inf_temp_0_008, h_68b387a1_timeout_precalc_true_acc_rt_weight_inf_temp_0_008, label="temp=0.008")
        #plt.plot(h_68b387a1_timeout_precalc_runtime_frac_rt_weight_inf_temp_0_004, h_68b387a1_timeout_precalc_true_acc_rt_weight_inf_temp_0_004, label="temp=0.004")
        #plt.plot(h_1db3f33f_timeout_precalc_runtime_frac_rt_weight_inf_temp_1e_06, h_1db3f33f_timeout_precalc_true_acc_rt_weight_inf_temp_1e_06, label="temp=1e-06")
        #plt.plot(h_2573bdc1_timeout_precalc_runtime_frac_inf, h_2573bdc1_timeout_precalc_true_acc_inf, color="r", label="temp=0 (select best instance)")
        #plt.plot(h_2573bdc1_timeout_precalc_runtime_frac_inf, h_2573bdc1_timeout_precalc_cross_acc_inf, color="b", label="temp=0 (select best instance) cross acc")



        #plt.plot(h_4965603a_timeout_precalc_runtime_frac_100, h_4965603a_timeout_precalc_true_acc_100, label="true_acc; rt_weight=100")
        #plt.plot(h_4965603a_timeout_precalc_runtime_frac_100, h_4965603a_timeout_precalc_cross_acc_100, label="cross_acc; rt_weight=100")

        #plt.plot(h_4965603a_timeout_precalc_runtime_frac_0_01, h_4965603a_timeout_precalc_true_acc_0_01, label="true_acc; rt_weight=0.01")
        #plt.plot(h_4965603a_timeout_precalc_runtime_frac_0_01, h_4965603a_timeout_precalc_cross_acc_0_01, label="cross_acc;rt_weight=0.01")


        #plt.plot(h_9d7f25f4_timeout_precalc_runtime_frac_0_01, h_9d7f25f4_timeout_precalc_true_acc_0_01, label="rt_weight=0.01_break after 0.4, true accuracy precalculation")
        #plt.plot(h_9d7f25f4_choose_instances_random_runtime_frac_0_01, h_9d7f25f4_choose_instances_random_true_acc_0_01, label="rt_weight=0.01_break after 0.4, random sel")
        #plt.plot(h_9d7f25f4_variance_based_selection_1_runtime_frac_0_01, h_9d7f25f4_variance_based_selection_1_true_acc_0_01, label="rt_weight=0.01_break after 0.4, variance based sel")

        #plt.plot(h_3a8224d8_choose_instances_random_runtime_frac_rt_weight_1_temp_None, h_3a8224d8_choose_instances_random_true_acc_rt_weight_1_temp_None, label="select random")#label="choose variance based after min thresh")
        #plt.plot(h_3a8224d8_highest_rt_selection_runtime_frac_rt_weight_1_temp_None, h_3a8224d8_highest_rt_selection_true_acc_rt_weight_1_temp_None, label="select $argmax(R(i, τ_i))$")
        #plt.plot(h_3a8224d8_lowest_rt_selection_runtime_frac_rt_weight_1_temp_None, h_3a8224d8_lowest_rt_selection_true_acc_rt_weight_1_temp_None, label="select $argmin(R(i, τ_i))$")
        
        #plt.plot(h_e8252123_variance_based_selection_1_runtime_frac_rt_weight_1_temp_None, h_e8252123_variance_based_selection_1_true_acc_rt_weight_1_temp_None, label="select $argmax(Var(i))$")
        #plt.plot(h_3a8224d8_lowest_variance_runtime_frac_rt_weight_1_temp_None, h_3a8224d8_lowest_variance_true_acc_rt_weight_1_temp_None, label="select $argmin(Var(i))$")

        #plt.plot(h_3a8224d8_lowest_variances_per_rt_runtime_frac_rt_weight_1_temp_None, h_3a8224d8_lowest_variances_per_rt_true_acc_rt_weight_1_temp_None, label='select $argmin(Var(i)/R(i, τ_i))$')
        #plt.plot(h_3a8224d8_variance_based_selection_1_runtime_frac_rt_weight_1_temp_None, h_3a8224d8_variance_based_selection_1_true_acc_rt_weight_1_temp_None, label='select $argmin(Var(i)/R(i, τ_i))$')
        
        #knapsack
        #plt.plot(h_c6afc00c_variance_based_selection_1_runtime_frac_rt_weight_1_temp_None, h_c6afc00c_variance_based_selection_1_true_acc_rt_weight_1_temp_None, label="knapsack")
        #plt.plot(h_8e719502_choose_instances_random_runtime_frac_rt_weight_1_temp_None, h_8e719502_choose_instances_random_true_acc_rt_weight_1_temp_None, label="knapsack to 0.4 choose random")
        #plt.plot(h_8e719502_variance_based_selection_1_runtime_frac_rt_weight_1_temp_None, h_8e719502_variance_based_selection_1_true_acc_rt_weight_1_temp_None)

        plt.plot(al_low_delta_rt, al_high_delta_acc, marker='o', linestyle='None', label='active learning')
        plt.legend()
        plt.xlabel("Fraction of Runtime")
        plt.xlim(right=1)
        #plt.ylim(0, 1.05)
        plt.ylabel("Ranking Accuracy")
        plt.title("Comparision of different instance selection methods")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        # Show or save
        plt.show()
        #plt.savefig("instance_histogram.png", dpi=300)

    def plot_histogramm(self, df):
        # 3. Plot the histogram        
        print(df.iloc[1688])
        plt.figure(figsize=(10, 6))
        plt.hist(df.iloc[1688], bins='auto')
        plt.xlabel("sover included")
        plt.ylabel("Count")
        plt.title("Determined Thresholds without solver Kissat_MAB-HyWalk")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        # Show or save
        plt.show()
        #plt.savefig("instance_histogram.png", dpi=300) """

    def plot_regression(self, prediction, actual, label, color):
        # Reshape prediction to 2D for sklearn
        prediction_np = np.array(prediction).reshape(-1, 1)
        actual_np = np.array(actual)

        # Fit linear regression
        model = LinearRegression()
        model.fit(prediction_np, actual_np)
        predicted_line = model.predict(prediction_np)

        # Compute RMSE
        rmse = mean_squared_error(actual_np, predicted_line, squared=False)

        # Plot original data
        plt.plot(prediction, actual, marker='.', linestyle='None', label=f"{label} (RMSE={rmse:.2f})", color=color)

        # Plot regression line
        sorted_indices = np.argsort(prediction)
        plt.plot(np.array(prediction)[sorted_indices], predicted_line[sorted_indices], linestyle='-', color=color)

        return model.coef_[0], model.intercept_, rmse

    def visualize_predictions(self, df_rated: pd.DataFrame):

        par_2_scores_series = df_rated.mean(axis=0)
        print(par_2_scores_series)

        df_rated_4000 = df_rated
        df_rated_4000[df_rated_4000 > 4000] = 8000
        par_2_scores_series_4000 = df_rated_4000.mean(axis=0)

        df_rated_3000 = df_rated
        df_rated_3000[df_rated_3000 > 3000] = 6000
        par_2_scores_series_3000 = df_rated_3000.mean(axis=0)

        df_rated_2000 = df_rated
        df_rated_2000[df_rated_2000 > 2000] = 4000
        par_2_scores_series_2000 = df_rated_2000.mean(axis=0)

        df_rated_1000 = df_rated
        df_rated_1000[df_rated_1000 > 1000] = 2000
        par_2_scores_series_1000 = df_rated_1000.mean(axis=0)

        actual = []
        prediction_4000 = []
        prediction_3000 = []
        prediction_2000 = []
        prediction_1000 = []
        for solver in par_2_scores_series.index:
            actual.append(par_2_scores_series[solver])
            prediction_4000.append(par_2_scores_series_4000[solver])
            prediction_3000.append(par_2_scores_series_3000[solver])
            prediction_2000.append(par_2_scores_series_2000[solver])
            prediction_1000.append(par_2_scores_series_1000[solver])

        plt.figure(figsize=(10, 6))
        #plt.plot(actual, actual, marker=".", linestyle='None', label="timeout=5000")
        colors = ['blue', 'orange', 'green', 'red', 'black']
        m5000, c5000, rmse5000 = self.plot_regression(actual, actual, "timeout=5000", colors[0])
        m4000, c4000, rmse4000 = self.plot_regression(prediction_4000, actual, "timeout=4000", colors[1])
        m3000, c3000, rmse3000 = self.plot_regression(prediction_3000, actual, "timeout=3000", colors[2])
        m2000, c2000, rmse2000 = self.plot_regression(prediction_2000, actual, "timeout=2000", colors[3])
        m1000, c1000, rmse1000 = self.plot_regression(prediction_1000, actual, "timeout=1000", colors[4])
        plt.legend()
        plt.xlabel("par-2-score with timeout < 5000")
        plt.ylabel("par-2-score with timeout = 5000")
        plt.title("Comparision of lower timeout par-2-scores")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        # Show or save
        plt.show()


    def visualize_predictions_exclude_instances(self, df_rated: pd.DataFrame, df_runtimes: pd.DataFrame):

        par_2_scores_series = df_rated.mean(axis=0)

        instance_mean_rt = df_runtimes.mean(axis=1)
        print(instance_mean_rt.sort_values())

        mask_500 = instance_mean_rt > 500
        df_rated_500 = df_runtimes.copy()
        df_rated_500[mask_500] = np.nan
        par_2_scores_500 = df_rated_500.mean(axis=0)

        mask_1000 = instance_mean_rt > 1000
        df_rated_1000 = df_runtimes.copy()
        df_rated_1000[mask_1000] = np.nan
        par_2_scores_1000 = df_rated_1000.mean(axis=0)

        mask_3000 = instance_mean_rt > 3000
        df_rated_3000 = df_runtimes.copy()
        df_rated_3000[mask_3000] = np.nan
        par_2_scores_3000 = df_rated_3000.mean(axis=0)

        mask_2000 = instance_mean_rt > 2000
        df_rated_2000 = df_runtimes.copy()
        df_rated_2000[mask_2000] = np.nan
        par_2_scores_2000 = df_rated_2000.mean(axis=0)

        mask_4000 = instance_mean_rt > 4000
        df_rated_4000 = df_runtimes.copy()
        df_rated_4000[mask_4000] = np.nan
        df_rated_4000[df_rated_4000 > 1000] = 2000
        par_2_scores_4000 = df_rated_4000.mean(axis=0)

        actual = []
        prediction_500 = []
        prediction_1000 = []
        prediction_3000 = []
        prediction_2000 = []
        prediction_4000 = []
        for solver in par_2_scores_series.index:
            actual.append(par_2_scores_series[solver])
            prediction_500.append(par_2_scores_500[solver])
            prediction_1000.append(par_2_scores_1000[solver])
            prediction_3000.append(par_2_scores_3000[solver])
            prediction_2000.append(par_2_scores_2000[solver])
            prediction_4000.append(par_2_scores_4000[solver])

        plt.figure(figsize=(10, 6))
        #plt.plot(actual, actual, marker=".", linestyle='None', label="timeout=5000")
        colors = ['blue', 'orange', 'green', 'red', 'black', 'purple', 'gray']
        #m100, c100, rmse100 = self.plot_regression(prediction_500, actual, "mean_time <= 500", colors[5])
        #m1000, c1000, rmse1000 = self.plot_regression(prediction_1000, actual, "mean_time <= 1000", colors[4])        
        #m3000, c3000, rmse3000 = self.plot_regression(prediction_2000, actual, "mean_time <= 2000", colors[2])
        #m2000, c2000, rmse2000 = self.plot_regression(prediction_3000, actual, "mean_time <= 3000", colors[3])
        m4000, c4000, rmse4000 = self.plot_regression(prediction_4000, actual, "mean_time <= 4000 and timeout = 1000", colors[1])
        m5000, c5000, rmse5000 = self.plot_regression(actual, actual, "mean_time <= inf", colors[0])
        
        plt.legend()
        plt.xlabel("mean_time limited")
        plt.ylabel("mean_time unlimited")
        plt.title("Comparision of reduced instance pool par-2-scores")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        # Show or save
        plt.show()