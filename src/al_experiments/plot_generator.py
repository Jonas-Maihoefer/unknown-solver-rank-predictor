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

    def plot_avg_results(self, df: pd.DataFrame, num_samples, breaking_name):
        # filter
        wanted_measurements = [f"determine_timeouts_{breaking_name}_true_acc_v2", f"determine_timeouts_{breaking_name}_cross_acc"]
        for sel_fn in self.exp_config.instance_selections:
            wanted_measurements.append(f"{sel_fn.__name__}_{breaking_name}_true_acc_v2")
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

        out_path = os.path.join(self.out_dir, f"average_results__{breaking_name}.png")

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
        self, all_results, solver_string, breaking_name
    ):
        # construct df
        df = pd.DataFrame.from_records(all_results)

        # filtering
        wanted_solver = [solver_string]
        df_sub = df[df["solver"].isin(wanted_solver)]
        wanted_measurements = [f"determine_timeouts_{breaking_name}_true_acc_v2", f"determine_timeouts_{breaking_name}_cross_acc"]
        for sel_fn in self.exp_config.instance_selections:
            wanted_measurements.append(f"{sel_fn.__name__}_{breaking_name}_true_acc_v2")
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

        out_path = os.path.join(self.out_dir, f"{solver_string}_{breaking_name}_results.png")

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

    def get_all_measurements_bottom_up(self, paths, plot_type, attribute_1=None, attribute_2=None):
        result_string = ""

        timeout_breaking_methods = ['until_cross_acc_0_96','until_cross_acc_0_97','until_cross_acc_0_98','until_cross_acc_0_99','until_cross_acc_0_965','until_cross_acc_0_975','until_cross_acc_0_985','until_cross_acc_0_995','until_cross_acc_1_00','until_cross_acc_0_96','until_cross_acc_0_97','until_cross_acc_0_98','until_cross_acc_0_99','until_cross_acc_0_965','until_cross_acc_0_975','until_cross_acc_0_985','until_cross_acc_0_995','until_cross_acc_1_00','until_cross_acc_0_96','until_cross_acc_0_97','until_cross_acc_0_98','until_cross_acc_0_99','until_cross_acc_0_965','until_cross_acc_0_975','until_cross_acc_0_985','until_cross_acc_0_995','until_cross_acc_1_00','until_cross_acc_0_96','until_cross_acc_0_97','until_cross_acc_0_98','until_cross_acc_0_99','until_cross_acc_0_965','until_cross_acc_0_975','until_cross_acc_0_985','until_cross_acc_0_995','until_cross_acc_1_00','until_stab_0_9800','until_stab_0_9825','until_stab_0_9850','until_stab_0_9875','until_stab_0_9900','until_stab_0_9925','until_stab_0_9950','until_stab_0_9975','until_stab_0_9800','until_stab_0_9825','until_stab_0_9850','until_stab_0_9875','until_stab_0_9900','until_stab_0_9925','until_stab_0_9950','until_stab_0_9975','until_stab_0_9800','until_stab_0_9825','until_stab_0_9850','until_stab_0_9875','until_stab_0_9900','until_stab_0_9925','until_stab_0_9950','until_stab_0_9975','until_stab_0_9800','until_stab_0_9825','until_stab_0_9850','until_stab_0_9875','until_stab_0_9900','until_stab_0_9925','until_stab_0_9950','until_stab_0_9975','until_stab_0_9800','until_stab_0_9825','until_stab_0_9850','until_stab_0_9875','until_stab_0_9900','until_stab_0_9925','until_stab_0_9950','until_stab_0_9975','until_stab_0_9800','until_stab_0_9825','until_stab_0_9850','until_stab_0_9875','until_stab_0_9900','until_stab_0_9925','until_stab_0_9950','until_stab_0_9975','until_stab_0_9800','until_stab_0_9825','until_stab_0_9850','until_stab_0_9875','until_stab_0_9900','until_stab_0_9925','until_stab_0_9950','until_stab_0_9975','until_stab_0_9800','until_stab_0_9825','until_stab_0_9850','until_stab_0_9875','until_stab_0_9900','until_stab_0_9925','until_stab_0_9950','until_stab_0_9975','until_stab_0_9800','until_stab_0_9825','until_stab_0_9850','until_stab_0_9875','until_stab_0_9900','until_stab_0_9925','until_stab_0_9950','until_stab_0_9975','until_stab_0_9800','until_stab_0_9825','until_stab_0_9850','until_stab_0_9875','until_stab_0_9900','until_stab_0_9925','until_stab_0_9950','until_stab_0_9975','until_stab_0_9800','until_stab_0_9825','until_stab_0_9850','until_stab_0_9875','until_stab_0_9900','until_stab_0_9925','until_stab_0_9950','until_stab_0_9975','until_stab_0_9800','until_stab_0_9825','until_stab_0_9850','until_stab_0_9875','until_stab_0_9900','until_stab_0_9925','until_stab_0_9950','until_stab_0_9975','until_stab_0_9800','until_stab_0_9825','until_stab_0_9850','until_stab_0_9875','until_stab_0_9900','until_stab_0_9925','until_stab_0_9950','until_stab_0_9975','until_stab_0_9800','until_stab_0_9825','until_stab_0_9850','until_stab_0_9875','until_stab_0_9900','until_stab_0_9925','until_stab_0_9950','until_stab_0_9975','until_stab_0_9800','until_stab_0_9825','until_stab_0_9850','until_stab_0_9875','until_stab_0_9900','until_stab_0_9925','until_stab_0_9950','until_stab_0_9975','until_cross_acc_0_96','until_cross_acc_0_97','until_cross_acc_0_98','until_cross_acc_0_99','until_cross_acc_0_965','until_cross_acc_0_975','until_cross_acc_0_985','until_cross_acc_0_995','until_cross_acc_1_00','until_cross_acc_0_96','until_cross_acc_0_97','until_cross_acc_0_98','until_cross_acc_0_99','until_cross_acc_0_965','until_cross_acc_0_975','until_cross_acc_0_985','until_cross_acc_0_995','until_cross_acc_1_00','until_cross_acc_0_96','until_cross_acc_0_97','until_cross_acc_0_98','until_cross_acc_0_99','until_cross_acc_0_965','until_cross_acc_0_975','until_cross_acc_0_985','until_cross_acc_0_995','until_cross_acc_1_00','until_cross_acc_0_96','until_cross_acc_0_97','until_cross_acc_0_98','until_cross_acc_0_99','until_cross_acc_0_965','until_cross_acc_0_975','until_cross_acc_0_985','until_cross_acc_0_995','until_cross_acc_1_00','until_cross_acc_0_96','until_cross_acc_0_97','until_cross_acc_0_98','until_cross_acc_0_99','until_cross_acc_0_965','until_cross_acc_0_975','until_cross_acc_0_985','until_cross_acc_0_995','until_cross_acc_1_00','until_cross_acc_0_96','until_cross_acc_0_97','until_cross_acc_0_98','until_cross_acc_0_99','until_cross_acc_0_965','until_cross_acc_0_975','until_cross_acc_0_985','until_cross_acc_0_995','until_cross_acc_1_00','until_cross_acc_0_96','until_cross_acc_0_97','until_cross_acc_0_98','until_cross_acc_0_99','until_cross_acc_0_965','until_cross_acc_0_975','until_cross_acc_0_985','until_cross_acc_0_995','until_cross_acc_1_00','until_cross_acc_0_96','until_cross_acc_0_97','until_cross_acc_0_98','until_cross_acc_0_99','until_cross_acc_0_965','until_cross_acc_0_975','until_cross_acc_0_985','until_cross_acc_0_995','until_cross_acc_1_00','until_cross_acc_0_96','until_cross_acc_0_97','until_cross_acc_0_98','until_cross_acc_0_99','until_cross_acc_0_965','until_cross_acc_0_975','until_cross_acc_0_985','until_cross_acc_0_995','until_cross_acc_1_00','until_cross_acc_0_96','until_cross_acc_0_97','until_cross_acc_0_98','until_cross_acc_0_99','until_cross_acc_0_965','until_cross_acc_0_975','until_cross_acc_0_985','until_cross_acc_0_995','until_cross_acc_1_00','until_cross_acc_0_96','until_cross_acc_0_97','until_cross_acc_0_98','until_cross_acc_0_99','until_cross_acc_0_965','until_cross_acc_0_975','until_cross_acc_0_985','until_cross_acc_0_995','until_cross_acc_1_00','until_cross_acc_0_96','until_cross_acc_0_97','until_cross_acc_0_98','until_cross_acc_0_99','until_cross_acc_0_965','until_cross_acc_0_975','until_cross_acc_0_985','until_cross_acc_0_995','until_cross_acc_1_00']
        timeout_breaking_names = ['$mathrm{fcp}_{geq 0.96}$','$mathrm{fcp}_{geq 0.97}$','$mathrm{fcp}_{geq 0.98}$','$mathrm{fcp}_{geq 0.99}$','$mathrm{fcp}_{geq 0.965}$','$mathrm{fcp}_{geq 0.975}$','$mathrm{fcp}_{geq 0.985}$','$mathrm{fcp}_{geq 0.995}$','$mathrm{fcp}_{geq 1.0}$','$mathrm{fcp}_{geq 0.96}$','$mathrm{fcp}_{geq 0.97}$','$mathrm{fcp}_{geq 0.98}$','$mathrm{fcp}_{geq 0.99}$','$mathrm{fcp}_{geq 0.965}$','$mathrm{fcp}_{geq 0.975}$','$mathrm{fcp}_{geq 0.985}$','$mathrm{fcp}_{geq 0.995}$','$mathrm{fcp}_{geq 1.0}$','$mathrm{fcp}_{geq 0.96}$','$mathrm{fcp}_{geq 0.97}$','$mathrm{fcp}_{geq 0.98}$','$mathrm{fcp}_{geq 0.99}$','$mathrm{fcp}_{geq 0.965}$','$mathrm{fcp}_{geq 0.975}$','$mathrm{fcp}_{geq 0.985}$','$mathrm{fcp}_{geq 0.995}$','$mathrm{fcp}_{geq 1.0}$','$mathrm{fcp}_{geq 0.96}$','$mathrm{fcp}_{geq 0.97}$','$mathrm{fcp}_{geq 0.98}$','$mathrm{fcp}_{geq 0.99}$','$mathrm{fcp}_{geq 0.965}$','$mathrm{fcp}_{geq 0.975}$','$mathrm{fcp}_{geq 0.985}$','$mathrm{fcp}_{geq 0.995}$','$mathrm{fcp}_{geq 1.0}$','$mathrm{stab}_{geq 0.98}$','$mathrm{stab}_{geq 0.9825}$','$mathrm{stab}_{geq 0.985}$','$mathrm{stab}_{geq 0.9875}$','$mathrm{stab}_{geq 0.99}$','$mathrm{stab}_{geq 0.9925}$','$mathrm{stab}_{geq 0.995}$','$mathrm{stab}_{geq 0.9975}$','$mathrm{stab}_{geq 0.98}$','$mathrm{stab}_{geq 0.9825}$','$mathrm{stab}_{geq 0.985}$','$mathrm{stab}_{geq 0.9875}$','$mathrm{stab}_{geq 0.99}$','$mathrm{stab}_{geq 0.9925}$','$mathrm{stab}_{geq 0.995}$','$mathrm{stab}_{geq 0.9975}$','$mathrm{stab}_{geq 0.98}$','$mathrm{stab}_{geq 0.9825}$','$mathrm{stab}_{geq 0.985}$','$mathrm{stab}_{geq 0.9875}$','$mathrm{stab}_{geq 0.99}$','$mathrm{stab}_{geq 0.9925}$','$mathrm{stab}_{geq 0.995}$','$mathrm{stab}_{geq 0.9975}$','$mathrm{stab}_{geq 0.98}$','$mathrm{stab}_{geq 0.9825}$','$mathrm{stab}_{geq 0.985}$','$mathrm{stab}_{geq 0.9875}$','$mathrm{stab}_{geq 0.99}$','$mathrm{stab}_{geq 0.9925}$','$mathrm{stab}_{geq 0.995}$','$mathrm{stab}_{geq 0.9975}$','$mathrm{stab}_{geq 0.98}$','$mathrm{stab}_{geq 0.9825}$','$mathrm{stab}_{geq 0.985}$','$mathrm{stab}_{geq 0.9875}$','$mathrm{stab}_{geq 0.99}$','$mathrm{stab}_{geq 0.9925}$','$mathrm{stab}_{geq 0.995}$','$mathrm{stab}_{geq 0.9975}$','$mathrm{stab}_{geq 0.98}$','$mathrm{stab}_{geq 0.9825}$','$mathrm{stab}_{geq 0.985}$','$mathrm{stab}_{geq 0.9875}$','$mathrm{stab}_{geq 0.99}$','$mathrm{stab}_{geq 0.9925}$','$mathrm{stab}_{geq 0.995}$','$mathrm{stab}_{geq 0.9975}$','$mathrm{stab}_{geq 0.98}$','$mathrm{stab}_{geq 0.9825}$','$mathrm{stab}_{geq 0.985}$','$mathrm{stab}_{geq 0.9875}$','$mathrm{stab}_{geq 0.99}$','$mathrm{stab}_{geq 0.9925}$','$mathrm{stab}_{geq 0.995}$','$mathrm{stab}_{geq 0.9975}$','$mathrm{stab}_{geq 0.98}$','$mathrm{stab}_{geq 0.9825}$','$mathrm{stab}_{geq 0.985}$','$mathrm{stab}_{geq 0.9875}$','$mathrm{stab}_{geq 0.99}$','$mathrm{stab}_{geq 0.9925}$','$mathrm{stab}_{geq 0.995}$','$mathrm{stab}_{geq 0.9975}$','$mathrm{stab}_{geq 0.98}$','$mathrm{stab}_{geq 0.9825}$','$mathrm{stab}_{geq 0.985}$','$mathrm{stab}_{geq 0.9875}$','$mathrm{stab}_{geq 0.99}$','$mathrm{stab}_{geq 0.9925}$','$mathrm{stab}_{geq 0.995}$','$mathrm{stab}_{geq 0.9975}$','$mathrm{stab}_{geq 0.98}$','$mathrm{stab}_{geq 0.9825}$','$mathrm{stab}_{geq 0.985}$','$mathrm{stab}_{geq 0.9875}$','$mathrm{stab}_{geq 0.99}$','$mathrm{stab}_{geq 0.9925}$','$mathrm{stab}_{geq 0.995}$','$mathrm{stab}_{geq 0.9975}$','$mathrm{stab}_{geq 0.98}$','$mathrm{stab}_{geq 0.9825}$','$mathrm{stab}_{geq 0.985}$','$mathrm{stab}_{geq 0.9875}$','$mathrm{stab}_{geq 0.99}$','$mathrm{stab}_{geq 0.9925}$','$mathrm{stab}_{geq 0.995}$','$mathrm{stab}_{geq 0.9975}$','$mathrm{stab}_{geq 0.98}$','$mathrm{stab}_{geq 0.9825}$','$mathrm{stab}_{geq 0.985}$','$mathrm{stab}_{geq 0.9875}$','$mathrm{stab}_{geq 0.99}$','$mathrm{stab}_{geq 0.9925}$','$mathrm{stab}_{geq 0.995}$','$mathrm{stab}_{geq 0.9975}$','$mathrm{stab}_{geq 0.98}$','$mathrm{stab}_{geq 0.9825}$','$mathrm{stab}_{geq 0.985}$','$mathrm{stab}_{geq 0.9875}$','$mathrm{stab}_{geq 0.99}$','$mathrm{stab}_{geq 0.9925}$','$mathrm{stab}_{geq 0.995}$','$mathrm{stab}_{geq 0.9975}$','$mathrm{stab}_{geq 0.98}$','$mathrm{stab}_{geq 0.9825}$','$mathrm{stab}_{geq 0.985}$','$mathrm{stab}_{geq 0.9875}$','$mathrm{stab}_{geq 0.99}$','$mathrm{stab}_{geq 0.9925}$','$mathrm{stab}_{geq 0.995}$','$mathrm{stab}_{geq 0.9975}$','$mathrm{stab}_{geq 0.98}$','$mathrm{stab}_{geq 0.9825}$','$mathrm{stab}_{geq 0.985}$','$mathrm{stab}_{geq 0.9875}$','$mathrm{stab}_{geq 0.99}$','$mathrm{stab}_{geq 0.9925}$','$mathrm{stab}_{geq 0.995}$','$mathrm{stab}_{geq 0.9975}$','$mathrm{fcp}_{geq 0.96}$','$mathrm{fcp}_{geq 0.97}$','$mathrm{fcp}_{geq 0.98}$','$mathrm{fcp}_{geq 0.99}$','$mathrm{fcp}_{geq 0.965}$','$mathrm{fcp}_{geq 0.975}$','$mathrm{fcp}_{geq 0.985}$','$mathrm{fcp}_{geq 0.995}$','$mathrm{fcp}_{geq 1.0}$','$mathrm{fcp}_{geq 0.96}$','$mathrm{fcp}_{geq 0.97}$','$mathrm{fcp}_{geq 0.98}$','$mathrm{fcp}_{geq 0.99}$','$mathrm{fcp}_{geq 0.965}$','$mathrm{fcp}_{geq 0.975}$','$mathrm{fcp}_{geq 0.985}$','$mathrm{fcp}_{geq 0.995}$','$mathrm{fcp}_{geq 1.0}$','$mathrm{fcp}_{geq 0.96}$','$mathrm{fcp}_{geq 0.97}$','$mathrm{fcp}_{geq 0.98}$','$mathrm{fcp}_{geq 0.99}$','$mathrm{fcp}_{geq 0.965}$','$mathrm{fcp}_{geq 0.975}$','$mathrm{fcp}_{geq 0.985}$','$mathrm{fcp}_{geq 0.995}$','$mathrm{fcp}_{geq 1.0}$','$mathrm{fcp}_{geq 0.96}$','$mathrm{fcp}_{geq 0.97}$','$mathrm{fcp}_{geq 0.98}$','$mathrm{fcp}_{geq 0.99}$','$mathrm{fcp}_{geq 0.965}$','$mathrm{fcp}_{geq 0.975}$','$mathrm{fcp}_{geq 0.985}$','$mathrm{fcp}_{geq 0.995}$','$mathrm{fcp}_{geq 1.0}$','$mathrm{fcp}_{geq 0.96}$','$mathrm{fcp}_{geq 0.97}$','$mathrm{fcp}_{geq 0.98}$','$mathrm{fcp}_{geq 0.99}$','$mathrm{fcp}_{geq 0.965}$','$mathrm{fcp}_{geq 0.975}$','$mathrm{fcp}_{geq 0.985}$','$mathrm{fcp}_{geq 0.995}$','$mathrm{fcp}_{geq 1.0}$','$mathrm{fcp}_{geq 0.96}$','$mathrm{fcp}_{geq 0.97}$','$mathrm{fcp}_{geq 0.98}$','$mathrm{fcp}_{geq 0.99}$','$mathrm{fcp}_{geq 0.965}$','$mathrm{fcp}_{geq 0.975}$','$mathrm{fcp}_{geq 0.985}$','$mathrm{fcp}_{geq 0.995}$','$mathrm{fcp}_{geq 1.0}$','$mathrm{fcp}_{geq 0.96}$','$mathrm{fcp}_{geq 0.97}$','$mathrm{fcp}_{geq 0.98}$','$mathrm{fcp}_{geq 0.99}$','$mathrm{fcp}_{geq 0.965}$','$mathrm{fcp}_{geq 0.975}$','$mathrm{fcp}_{geq 0.985}$','$mathrm{fcp}_{geq 0.995}$','$mathrm{fcp}_{geq 1.0}$','$mathrm{fcp}_{geq 0.96}$','$mathrm{fcp}_{geq 0.97}$','$mathrm{fcp}_{geq 0.98}$','$mathrm{fcp}_{geq 0.99}$','$mathrm{fcp}_{geq 0.965}$','$mathrm{fcp}_{geq 0.975}$','$mathrm{fcp}_{geq 0.985}$','$mathrm{fcp}_{geq 0.995}$','$mathrm{fcp}_{geq 1.0}$','$mathrm{fcp}_{geq 0.96}$','$mathrm{fcp}_{geq 0.97}$','$mathrm{fcp}_{geq 0.98}$','$mathrm{fcp}_{geq 0.99}$','$mathrm{fcp}_{geq 0.965}$','$mathrm{fcp}_{geq 0.975}$','$mathrm{fcp}_{geq 0.985}$','$mathrm{fcp}_{geq 0.995}$','$mathrm{fcp}_{geq 1.0}$','$mathrm{fcp}_{geq 0.96}$','$mathrm{fcp}_{geq 0.97}$','$mathrm{fcp}_{geq 0.98}$','$mathrm{fcp}_{geq 0.99}$','$mathrm{fcp}_{geq 0.965}$','$mathrm{fcp}_{geq 0.975}$','$mathrm{fcp}_{geq 0.985}$','$mathrm{fcp}_{geq 0.995}$','$mathrm{fcp}_{geq 1.0}$','$mathrm{fcp}_{geq 0.96}$','$mathrm{fcp}_{geq 0.97}$','$mathrm{fcp}_{geq 0.98}$','$mathrm{fcp}_{geq 0.99}$','$mathrm{fcp}_{geq 0.965}$','$mathrm{fcp}_{geq 0.975}$','$mathrm{fcp}_{geq 0.985}$','$mathrm{fcp}_{geq 0.995}$','$mathrm{fcp}_{geq 1.0}$','$mathrm{fcp}_{geq 0.96}$','$mathrm{fcp}_{geq 0.97}$','$mathrm{fcp}_{geq 0.98}$','$mathrm{fcp}_{geq 0.99}$','$mathrm{fcp}_{geq 0.965}$','$mathrm{fcp}_{geq 0.975}$','$mathrm{fcp}_{geq 0.985}$','$mathrm{fcp}_{geq 0.995}$','$mathrm{fcp}_{geq 1.0}$']
        selection_methods = ['choose_instances_random', 'variance_based_selection_1', 'highest_rt_selection', 'lowest_variance', 'highest_variance', 'lowest_variances_per_rt', 'lowest_rt_selection']
        selection_names = ['$\mathrm{rand}$', '$\mathrm{max}_{V/R}$', '$\mathrm{max}_{R}$', '$\mathrm{min}_{V}$', '$\mathrm{max}_{V}$', '$\mathrm{min}_{V/R}$', '$\mathrm{min}_{R}$']
        filterings = [False               ,False               ,False               ,False               ,False               ,False               ,False               ,False               ,False               ,True             ,True             ,True             ,True             ,True             ,True             ,True             ,True             ,True             ,True        ,True        ,True        ,True        ,True        ,True        ,True        ,True        ,True        ,False          ,False          ,False          ,False          ,False          ,False          ,False          ,False          ,False          ,True              ,True              ,True              ,True              ,True              ,True              ,True              ,True              ,False                ,False                ,False                ,False                ,False                ,False                ,False                ,False                ,True                   ,True                   ,True                   ,True                   ,True                   ,True                   ,True                   ,True                   ,False                     ,False                     ,False                     ,False                     ,False                     ,False                     ,False                     ,False                     ,False               ,False               ,False               ,False               ,False               ,False               ,False               ,False               ,True        ,True        ,True        ,True        ,True        ,True        ,True        ,True        ,False          ,False          ,False          ,False          ,False          ,False          ,False          ,False          ,True             ,True             ,True             ,True             ,True             ,True             ,True             ,True             ,False               ,False               ,False               ,False               ,False               ,False               ,False               ,False               ,True        ,True        ,True        ,True        ,True        ,True        ,True        ,True        ,False          ,False          ,False          ,False          ,False          ,False          ,False          ,False          ,True            ,True            ,True            ,True            ,True            ,True            ,True            ,True            ,False              ,False              ,False              ,False              ,False              ,False              ,False              ,False              ,True       ,True       ,True       ,True       ,True       ,True       ,True       ,True       ,False         ,False         ,False         ,False         ,False         ,False         ,False         ,False         ,True       ,True       ,True       ,True       ,True       ,True       ,True       ,True       ,True       ,False         ,False         ,False         ,False         ,False         ,False         ,False         ,False         ,False         ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,False    ,False    ,False    ,False    ,False    ,False    ,False    ,False    ,False    ,True       ,True       ,True       ,True       ,True       ,True       ,True       ,True       ,True       ,False         ,False         ,False         ,False         ,False         ,False         ,False         ,False         ,False         ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,False    ,False    ,False    ,False    ,False    ,False    ,False    ,False    ,False    ,True      ,True      ,True      ,True      ,True      ,True      ,True      ,True      ,True      ,False        ,False        ,False        ,False        ,False        ,False        ,False        ,False        ,False        ,True ,True ,True ,True ,True ,True ,True ,True ,True ,False   ,False   ,False   ,False   ,False   ,False   ,False   ,False   ,False] 
        t_instance_selections=['$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{best}$'  ,'$\mathrm{top}_3$' ,'$\mathrm{top}_3$' ,'$\mathrm{top}_3$' ,'$\mathrm{top}_3$' ,'$\mathrm{top}_3$' ,'$\mathrm{top}_3$' ,'$\mathrm{top}_3$' ,'$\mathrm{top}_3$' ,'$\mathrm{top}_3$' ,'$\mathrm{top}_3$' ,'$\mathrm{top}_3$' ,'$\mathrm{top}_3$' ,'$\mathrm{top}_3$' ,'$\mathrm{top}_3$' ,'$\mathrm{top}_3$' ,'$\mathrm{top}_3$' ,'$\mathrm{top}_3$' ,'$\mathrm{top}_3$' ,'$\mathrm{top}_3$' ,'$\mathrm{top}_3$' ,'$\mathrm{top}_3$' ,'$\mathrm{top}_3$' ,'$\mathrm{top}_3$' ,'$\mathrm{top}_3$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_3$' ,'$\mathrm{top}_3$' ,'$\mathrm{top}_3$' ,'$\mathrm{top}_3$' ,'$\mathrm{top}_3$' ,'$\mathrm{top}_3$' ,'$\mathrm{top}_3$' ,'$\mathrm{top}_3$' ,'$\mathrm{top}_3$' ,'$\mathrm{top}_3$' ,'$\mathrm{top}_3$' ,'$\mathrm{top}_3$' ,'$\mathrm{top}_3$' ,'$\mathrm{top}_3$' ,'$\mathrm{top}_3$' ,'$\mathrm{top}_3$' ,'$\mathrm{top}_3$' ,'$\mathrm{top}_3$' ,'$\mathrm{top}_3$' ,'$\mathrm{top}_3$' ,'$\mathrm{top}_3$' ,'$\mathrm{top}_3$' ,'$\mathrm{top}_3$' ,'$\mathrm{top}_3$' ,'$\mathrm{top}_3$' ,'$\mathrm{top}_3$' ,'$\mathrm{top}_3$' ,'$\mathrm{top}_3$' ,'$\mathrm{top}_3$' ,'$\mathrm{top}_3$' ,'$\mathrm{top}_3$' ,'$\mathrm{top}_3$' ,'$\mathrm{top}_3$' ,'$\mathrm{top}_3$' ,'$\mathrm{top}_3$' ,'$\mathrm{top}_3$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_9$' ,'$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$','$\mathrm{top}_27$']
        optimizations = ['$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^3$', '$u_i^3$', '$u_i^3$', '$u_i^3$', '$u_i^3$', '$u_i^3$', '$u_i^3$', '$u_i^3$', '$u_i^3$', '$u_i^3$', '$u_i^3$', '$u_i^3$', '$u_i^3$', '$u_i^3$', '$u_i^3$', '$u_i^3$', '$u_i^3$', '$u_i^3$', '$u_i^1$', '$u_i^1$', '$u_i^1$', '$u_i^1$', '$u_i^1$', '$u_i^1$', '$u_i^1$', '$u_i^1$', '$u_i^1$', '$u_i^1$', '$u_i^1$', '$u_i^1$', '$u_i^1$', '$u_i^1$', '$u_i^1$', '$u_i^1$', '$u_i^2$', '$u_i^2$', '$u_i^2$', '$u_i^2$', '$u_i^2$', '$u_i^2$', '$u_i^2$', '$u_i^2$', '$u_i^2$', '$u_i^2$', '$u_i^2$', '$u_i^2$', '$u_i^2$', '$u_i^2$', '$u_i^2$', '$u_i^2$', '$u_i^2$', '$u_i^2$', '$u_i^2$', '$u_i^2$', '$u_i^2$', '$u_i^2$', '$u_i^2$', '$u_i^2$', '$u_i^1$', '$u_i^1$', '$u_i^1$', '$u_i^1$', '$u_i^1$', '$u_i^1$', '$u_i^1$', '$u_i^1$', '$u_i^1$', '$u_i^1$', '$u_i^1$', '$u_i^1$', '$u_i^1$', '$u_i^1$', '$u_i^1$', '$u_i^1$', '$u_i^2$', '$u_i^2$', '$u_i^2$', '$u_i^2$', '$u_i^2$', '$u_i^2$', '$u_i^2$', '$u_i^2$', '$u_i^2$', '$u_i^2$', '$u_i^2$', '$u_i^2$', '$u_i^2$', '$u_i^2$', '$u_i^2$', '$u_i^2$', '$u_i^1$', '$u_i^1$', '$u_i^1$', '$u_i^1$', '$u_i^1$', '$u_i^1$', '$u_i^1$', '$u_i^1$', '$u_i^1$', '$u_i^1$', '$u_i^1$', '$u_i^1$', '$u_i^1$', '$u_i^1$', '$u_i^1$', '$u_i^1$', '$u_i^2$', '$u_i^2$', '$u_i^2$', '$u_i^2$', '$u_i^2$', '$u_i^2$', '$u_i^2$', '$u_i^2$', '$u_i^2$', '$u_i^2$', '$u_i^2$', '$u_i^2$', '$u_i^2$', '$u_i^2$', '$u_i^2$', '$u_i^2$', '$u_i^1$', '$u_i^1$', '$u_i^1$', '$u_i^1$', '$u_i^1$', '$u_i^1$', '$u_i^1$', '$u_i^1$', '$u_i^1$', '$u_i^1$', '$u_i^1$', '$u_i^1$', '$u_i^1$', '$u_i^1$', '$u_i^1$', '$u_i^1$', '$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^3$', '$u_i^3$', '$u_i^3$', '$u_i^3$', '$u_i^3$', '$u_i^3$', '$u_i^3$', '$u_i^3$', '$u_i^3$', '$u_i^3$', '$u_i^3$', '$u_i^3$', '$u_i^3$', '$u_i^3$', '$u_i^3$', '$u_i^3$', '$u_i^3$', '$u_i^3$', '$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^3$', '$u_i^3$', '$u_i^3$', '$u_i^3$', '$u_i^3$', '$u_i^3$', '$u_i^3$', '$u_i^3$', '$u_i^3$', '$u_i^3$', '$u_i^3$', '$u_i^3$', '$u_i^3$', '$u_i^3$', '$u_i^3$', '$u_i^3$', '$u_i^3$', '$u_i^3$', '$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^4$', '$u_i^3$', '$u_i^3$', '$u_i^3$', '$u_i^3$', '$u_i^3$', '$u_i^3$', '$u_i^3$', '$u_i^3$', '$u_i^3$', '$u_i^3$', '$u_i^3$', '$u_i^3$', '$u_i^3$', '$u_i^3$', '$u_i^3$', '$u_i^3$', '$u_i^3$', '$u_i^3$']
        timeout_breaking_methods.reverse()
        timeout_breaking_names.reverse()
        selection_names.reverse()
        filterings.reverse()
        t_instance_selections.reverse()
        optimizations.reverse()

        for path in paths:
            timeout_breaking = timeout_breaking_names.pop()
            timeout_breaking_method = timeout_breaking_methods.pop()
            filtering = filterings.pop()
            t_instance_selection = t_instance_selections.pop()
            optimization = optimizations.pop()
            selection_names_copy = selection_names.copy()
            print(path + ": " + timeout_breaking_method)
            df = pd.read_pickle(path, compression='gzip')
            for selection_method in selection_methods:
                selection_name = selection_names_copy.pop()
                print(f"sel_method={selection_method}; sel_name={selection_name}")
                instance_breaking = '$\mathrm{fcp}_{\geq 0.9}$'
                result_string += self.print_lowest_rf_cross_acc_greedy(df, timeout_breaking_method, selection_method, 0.9, f'{filtering} & {optimization} & {t_instance_selection} &  {timeout_breaking}  & {selection_name} & {instance_breaking}')
                instance_breaking = '$\mathrm{fcp}_{\geq 0.925}$'
                result_string += self.print_lowest_rf_cross_acc_greedy(df, timeout_breaking_method, selection_method, 0.925, f'{filtering} & {optimization} & {t_instance_selection} &  {timeout_breaking}  & {selection_name} & {instance_breaking}')
                instance_breaking = '$\mathrm{fcp}_{\geq 0.95}$'
                result_string += self.print_lowest_rf_cross_acc_greedy(df, timeout_breaking_method, selection_method, 0.95, f'{filtering} & {optimization} & {t_instance_selection} &  {timeout_breaking}  & {selection_name} & {instance_breaking}')
                instance_breaking = '$\mathrm{fcp}_{\geq 0.975}$'
                result_string += self.print_lowest_rf_cross_acc_greedy(df, timeout_breaking_method, selection_method, 0.975, f'{filtering} & {optimization} & {t_instance_selection} &  {timeout_breaking}  & {selection_name} & {instance_breaking}')

        print()
        print("combined:")
        print(result_string)

        results_df = pd.DataFrame(self.results, columns=["x", "y", "std_x", "std_y", "label"])
        results_df["method"] = "bottom-up"

        self.update_all_runs(results_df)

        if plot_type == 'pareto':
            pareto = self.pareto_front(results_df)
            self.plot_pareto_df(pareto, "Results for Bottom-Up Timeout Distribution")

        if plot_type == 'compare estimator':
            self.compare_estimator(results_df)

        if plot_type == 'compare':
            self.compare_attributes(results_df, attribute_1, attribute_2)

        self.results = []

    def get_all_measurements_random_timeout(self, paths, plot_type, attribute_1=None, attribute_2=None):
        """ result_string = ""

        filterings = [True, False]
        selection_methods = ['choose_instances_random', 'variance_based_selection_1', 'highest_rt_selection', 'lowest_variance', 'highest_variance', 'lowest_variances_per_rt', 'lowest_rt_selection']
        selection_names = ['$\mathrm{rand}$', '$\mathrm{max}_{V/R}$', '$\mathrm{max}_{R}$', '$\mathrm{min}_{V}$', '$\mathrm{max}_{V}$', '$\mathrm{min}_{V/R}$', '$\mathrm{min}_{R}$']

        filterings.reverse()
        selection_names.reverse()

        for path in paths:
            df = pd.read_pickle(path, compression='gzip')
            filtering = filterings.pop()
            selection_names_copy = selection_names.copy()
            for selection_method in selection_methods:
                selection_name = selection_names_copy.pop()
                print(f"sel_method={selection_method}; sel_name={selection_name}")
                breaking_condition = '$\mathrm{fcp}_{\geq 0.9}$'
                result_string += self.print_lowest_rf_cross_acc_greedy(df, 'not_needed', selection_method, 0.9, f'{filtering} & {selection_name} & {breaking_condition}')
                breaking_condition = '$\mathrm{fcp}_{\geq 0.925}$'
                result_string += self.print_lowest_rf_cross_acc_greedy(df, 'not_needed', selection_method, 0.925, f'{filtering} & {selection_name} & {breaking_condition}')
                breaking_condition = '$\mathrm{fcp}_{\geq 0.95}$'
                result_string += self.print_lowest_rf_cross_acc_greedy(df, 'not_needed', selection_method, 0.95, f'{filtering} & {selection_name} & {breaking_condition}')
                breaking_condition = '$\mathrm{fcp}_{\geq 0.975}$'
                result_string += self.print_lowest_rf_cross_acc_greedy(df, 'not_needed', selection_method, 0.975, f'{filtering} & {selection_name} & {breaking_condition} \\')

        print()
        print("combined:")
        print(result_string)

        results_df = pd.DataFrame(self.results, columns=["x", "y", "std_x", "std_y", "label"])
        results_df["method"] = "random" """

        stored_df = pd.read_pickle("./pickle/all-runs.pkl.gz", compression="gzip")

        results_df = stored_df[stored_df["method"] == "random"]

        if plot_type == 'pareto':
            pareto = self.pareto_front(results_df)
            self.plot_pareto_df(pareto, "Results for Random Timeout Distribution")

        if plot_type == 'compare estimator':
            self.compare_estimator(results_df)

        if plot_type == 'compare':
            self.compare_attributes(results_df, attribute_1, attribute_2)

        self.results = []

    def get_all_measurements_instance_wise(self, dfs, plot_type, attribute_1=None, attribute_2=None):
        """ result_string = ""

        deltas = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        filterings = [True, True, True, True, True, True, False, False, False, False, False, False]
        selection_methods = ['choose_instances_random', 'variance_based_selection_1', 'highest_rt_selection', 'lowest_variance', 'highest_variance', 'lowest_variances_per_rt', 'lowest_rt_selection']
        selection_names = ['$\mathrm{rand}$', '$\mathrm{max}_{V/R}$', '$\mathrm{max}_{R}$', '$\mathrm{min}_{V}$', '$\mathrm{max}_{V}$', '$\mathrm{min}_{V/R}$', '$\mathrm{min}_{R}$']

        deltas.reverse()
        filterings.reverse()
        selection_names.reverse()

        for df in dfs:
            delta = deltas.pop()
            filtering = filterings.pop()
            selection_names_copy = selection_names.copy()
            for selection_method in selection_methods:
                selection_name = selection_names_copy.pop()
                print(f"sel_method={selection_method}; sel_name={selection_name}")
                breaking_condition = '$\mathrm{fcp}_{\geq 0.9}$'
                result_string += self.print_lowest_rf_cross_acc(df, selection_method, 0.9, f'{filtering} & {delta} & {selection_name} & {breaking_condition}')
                breaking_condition = '$\mathrm{fcp}_{\geq 0.925}$'
                result_string += self.print_lowest_rf_cross_acc(df, selection_method, 0.925, f'{filtering} & {delta} & {selection_name} & {breaking_condition}')
                breaking_condition = '$\mathrm{fcp}_{\geq 0.95}$'
                result_string += self.print_lowest_rf_cross_acc(df, selection_method, 0.95, f'{filtering} & {delta} & {selection_name} & {breaking_condition}')
                breaking_condition = '$\mathrm{fcp}_{\geq 0.975}$'
                result_string += self.print_lowest_rf_cross_acc(df, selection_method, 0.975, f'{filtering} & {delta} & {selection_name} & {breaking_condition} \\')

        print()
        print("combined:")
        print(result_string)

        results_df = pd.DataFrame(self.results, columns=["x", "y", "std_x", "std_y", "label"])
        results_df["method"] = "instance-wise" """

        stored_df = pd.read_pickle("./pickle/all-runs.pkl.gz", compression="gzip")

        results_df = stored_df[stored_df["method"] == "instance-wise"]

        if plot_type == 'pareto':
            pareto = self.pareto_front(results_df)
            self.plot_pareto_df(pareto, "Results for Instance-Wise Timeout Distribution")

        if plot_type == 'compare estimator':
            self.compare_estimator(results_df)

        if plot_type == 'compare':
            self.compare_attributes(results_df, attribute_1, attribute_2)

        self.results = []

    def update_all_runs(self, append_df):
        path = "./pickle/all-runs.pkl.gz"
        stored_df = pd.read_pickle(path, compression="gzip")
        combined = pd.concat([stored_df, append_df], ignore_index=True)
        combined.to_pickle(path, compression="gzip")

    def compare_attributes(self, df, attribute_1, attribute_2):
        att_1 = df[df["label"].str.contains(attribute_1, regex=False, na=False)]
        if attribute_2 is None:
            att_2 = df[~df["label"].str.contains(attribute_1, regex=False, na=False)]
            attribute_2 = "not " + attribute_1
        else:
            att_2 = df[df["label"].str.contains(attribute_2, regex=False, na=False)]
        att_1_pareto = self.pareto_front(att_1)
        att_2_pareto = self.pareto_front(att_2)

        plt.figure(figsize=(8, 5))

        # Plot df_1
        sns.lineplot(
            data=att_1_pareto,
            x="x", y="y",
            drawstyle="steps-post",  # ensures horizontal, then vertical
            marker="o",
            label=attribute_1
        )

        # Plot df_2
        sns.lineplot(
            data=att_2_pareto,
            x="x", y="y",
            drawstyle="steps-post",
            marker="o",
            label=attribute_2
        )

        plt.xlabel("$\overline{O}_{\mathrm{rt}}$")
        #plt.xlim(right=1)
        #plt.ylim(0, 1.05)
        plt.ylabel("$\overline{O}_{\mathrm{acc}}$")
        plt.title("Compare Random Instance Selection to Non-Random Selection")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        plt.legend()
        plt.show()

    def compare_estimator(self, results_df):
        s_b_points = results_df[results_df['label'].str.contains('$s_\mathcal{B}$', regex=False)]
        s_fitted_points = results_df[results_df['label'].str.contains('$s_\mathrm{fitted}$', regex=False)]

        # Calculate the mean for the '$s_\mathcal{B}$' points
        mean_x_sb = s_b_points['x'].mean()
        mean_y_sb = s_b_points['y'].mean()

        # Calculate the mean for the '$s_\mathrm{fitted}$' points
        mean_x_fitted = s_fitted_points['x'].mean()
        mean_y_fitted = s_fitted_points['y'].mean()

        # Print the results
        print("Mean values for '$s_\\mathcal{B}$' points:")
        print(f"  Mean of x: {mean_x_sb}")
        print(f"  Mean of y: {mean_y_sb}")
        print("\\n" + "="*35 + "\\n") # Separator
        print("Mean values for '$s_\\mathrm{fitted}$' points:")
        print(f"  Mean of x: {mean_x_fitted}")
        print(f"  Mean of y: {mean_y_fitted}")

        # Create a new figure for the plot
        plt.figure(figsize=(8, 6))

        # Plot the points for '$s_\mathcal{B}$' in blue
        plt.scatter(s_b_points['x'], s_b_points['y'], color='blue', label='$s_\\mathcal{B}$')

        # Plot the points for '$s_\mathrm{fitted}$' in red
        plt.scatter(s_fitted_points['x'], s_fitted_points['y'], color='red', label='$s_\\mathrm{fitted}$')
        plt.legend()
        plt.xlabel("$\overline{O}_{\mathrm{rt}}$")
        #plt.xlim(right=1)
        #plt.ylim(0, 1.05)
        plt.ylabel("$\overline{O}_{\mathrm{acc}}$")
        plt.title("Comparison of Estimator Results")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        # Show or save
        plt.show()

        # Store the results and labels for the bar chart
        bar_labels = []
        metric_values = []

        # Helper function to perform the calculation
        def calculate_metric(sb_df, fitted_df):
            """Calculates the metric, returns NaN if either dataframe is empty."""
            if sb_df.empty or fitted_df.empty:
                return np.nan
            mean_y_sb = sb_df['y'].mean()
            mean_y_fitted = fitted_df['y'].mean()
            return (mean_y_sb - mean_y_fitted) * 100

        def process_slice(s_b_slice_df, s_fitted_full_df):
            """Finds corresponding points and calculates the metric."""
            if s_b_slice_df.empty:
                return np.nan

            # Get the s_B labels from the current slice
            s_b_labels_in_slice = s_b_slice_df['label']

            # Transform them to the corresponding s_fitted labels
            target_fitted_labels = s_b_labels_in_slice.str.replace(
                '$s_\mathcal{B}$', '$s_\mathrm{fitted}$', regex=False
            )

            # Find the corresponding s_fitted points from the *entire* s_fitted dataset
            corresponding_fitted_slice = s_fitted_full_df[
                s_fitted_full_df['label'].isin(target_fitted_labels)
            ]

            # Calculate the metric with the matched data
            return calculate_metric(s_b_slice_df, corresponding_fitted_slice)

        # ---- Calculation for y < 0.9 ----
        label = 'y < 0.9'
        s_b_slice = s_b_points[s_b_points['y'] < 0.9]
        bar_labels.append(label)
        metric_values.append(process_slice(s_b_slice, s_fitted_points))


        # ---- Loop for intervals between 0.9 and 1.0 ----
        for i in np.arange(0.90, 1.0, 0.01):
            lower_bound = i
            upper_bound = i + 0.01
            label = f'{lower_bound:.2f} - {upper_bound:.2f}'

            # Filter s_B points for the current interval
            s_b_slice = s_b_points[(s_b_points['y'] >= lower_bound) & (s_b_points['y'] < upper_bound)]

            bar_labels.append(label)
            metric_values.append(process_slice(s_b_slice, s_fitted_points))

        # --- 3. Generate the Bar Graph ---

        plt.figure(figsize=(12, 7)) # Create a larger figure to fit labels

        # Create the bar plot
        plt.bar(bar_labels, metric_values, color='skyblue')

        # Add titles and labels for clarity
        plt.xlabel("$\overline{O}_\mathrm{acc}$ using the $s_\mathrm{fitted}$-estimator")
        plt.ylabel("Improvment of $\overline{O}_\mathrm{acc}$ in %")
        plt.title("Improvment When Using $s_\mathrm{fitted}$ Instead of $s_\mathcal{B}$")
        plt.xticks(rotation=45, ha='right') # Rotate x-axis labels to prevent overlap
        plt.axhline(0, color='grey', linewidth=0.8) # Add a zero line for reference
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout() # Adjust plot to ensure everything fits

        # Display the plot
        plt.show()

    def plot_pareto_df(self, df, title):
        palette = sns.color_palette("tab20", n_colors=len(df))
        sns.set_theme(style="whitegrid")
        configuration = 1

        # Plot each label group
        for idx, row in df.iterrows():
            config_path = '\\raisebox{-0.2\\height}{\\includegraphics[width=0.6cm]{images/label_marker/config_' + str(configuration) + '.png}}'
            color = palette[idx % len(palette)]
            h = plt.errorbar(
                row["x"], row["y"],
                xerr=row["std_x"], yerr=row["std_y"],
                fmt="o", capsize=4, color=color
            )
            print(config_path + ' & ' + row["label"] + f' & ${round(row["x"], 3)} \pm {round(row["std_x"], 3)}$ & ${round(row["y"], 3)} \pm {round(row["std_y"], 3)}$ \\\\')
            configuration += 1
        plt.xlabel("$\overline{O}_{\mathrm{rt}}$")
        #plt.xlim(right=1)
        #plt.ylim(0, 1.05)
        plt.ylabel("$\overline{O}_{\mathrm{acc}}$")
        plt.title(title)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        # Show or save
        #plt.show()

    def save_errorbar_icons(self, n, filename_prefix="config"):
        palette = sns.color_palette("tab20", n_colors=n)

        for i, color in enumerate(palette):
            fig, ax = plt.subplots(figsize=(0.6, 0.6))
            ax.axis("off")

            # Bigger dot: markersize=10 (adjust as you like)
            ax.errorbar(
                0.5, 0.5,
                xerr=0.1, yerr=0.1,
                fmt="o", capsize=10, color=color,
                markersize=17,  # <-- make the central dot bigger
                linewidth=4,
                capthick=3
            )

            plt.savefig(f"{filename_prefix}_{i+1}.png",
                        dpi=200, bbox_inches="tight", transparent=True)
            plt.close(fig)

    def print_lowest_rf_cross_acc_greedy(self, df: pd.DataFrame, name, wanted_measurement, threshold: float, label) -> pd.DataFrame:
        """
        For each solver, find the smallet runtime_fraction where cross_acc > threshold.
        Also print the corresponding true_acc at that runtime_fraction.
        Additionally, compute and print averages and std deviations across solvers.
        """
        # Keep only cross_acc rows
        cross = df.loc[(df["measurement"] == f"{wanted_measurement}_{name}_cross_acc") & df["value"].notna(),
                       ["solver", "runtime_fraction", "value"]] \
                  .rename(columns={"value": "cross_acc"})

        true_v1 = df.loc[(df["measurement"] == f"{wanted_measurement}_{name}_true_acc_v1") & df["value"].notna(),
                      ["solver", "runtime_fraction", "value"]] \
                 .rename(columns={"value": "true_acc_v1"})

        true_v2 = df.loc[(df["measurement"] == f"{wanted_measurement}_{name}_true_acc_v2") & df["value"].notna(),
                      ["solver", "runtime_fraction", "value"]] \
                 .rename(columns={"value": "true_acc_v2"})

        solvers = df['solver'].unique()
        records = []

        for solver in solvers:
            cross_s = cross[cross['solver'] == solver]
            true_s_v1 = true_v1[true_v1['solver'] == solver]
            true_s_v2 = true_v2[true_v2['solver'] == solver]

            if cross_s.empty or true_s_v1.empty or true_s_v2.empty:
                print(f"No data found for solver {solver} with measurement {wanted_measurement}_{name} and label {label}")
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

            #print(f"{solver}: runtime_fraction = {min_rf:.4f}, cross_acc = {cross_val:.4f}, true_acc_v1 = {true_val_v1:.4f},  true_acc_v2 = {true_val_v2:.4f}")

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
        self.results.append((avg_rf, avg_true_v1, std_rf, std_true_v1, label + ' & $s_\mathcal{B}$'))
        self.results.append((avg_rf, avg_true_v2, std_rf, std_true_v2, label + ' & $s_\mathrm{fitted}$'))

        return f"$0.4$ & {wanted_measurement} & ${threshold}$ & ${avg_rf:.3f} \pm {std_rf:.3f}$ & ${avg_true_v1:.3f} \pm {std_true_v1:.3f}$ & ${avg_true_v2:.3f} \pm {std_true_v2:.3f}$ \\\\\n\n"

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

            #print(f"{solver}: runtime_fraction = {min_rf:.4f}, cross_acc = {cross_val:.4f}, true_acc_v1 = {true_val_v1:.4f},  true_acc_v2 = {true_val_v2:.4f}")

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
        self.results.append((avg_rf, avg_true_v1, std_rf, std_true_v1, label + ' & $s_\mathcal{B}$'))
        self.results.append((avg_rf, avg_true_v2, std_rf, std_true_v2, label + ' & $s_\mathrm{fitted}$'))

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
        print("start")
        #self.save_errorbar_icons(20, filename_prefix="config")
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

        ###### INSTANCE WISE ########

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

        print("nothing imported")

        ########## bottom-up: knapsack select best idx #################
        knapsack_rmse_until_cross_acc_0_960_no_filter = "./pickle/1f513996_0_until_cross_acc_0_96_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_0_970_no_filter = "./pickle/1f513996_0_until_cross_acc_0_97_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_0_980_no_filter = "./pickle/1f513996_0_until_cross_acc_0_98_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_0_990_no_filter = "./pickle/1f513996_0_until_cross_acc_0_99_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_0_965_no_filter = "./pickle/1f513996_0_until_cross_acc_0_965_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_0_975_no_filter = "./pickle/1f513996_0_until_cross_acc_0_975_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_0_985_no_filter = "./pickle/1f513996_0_until_cross_acc_0_985_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_0_995_no_filter = "./pickle/1f513996_0_until_cross_acc_0_995_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_1_000_no_filter = "./pickle/1f513996_0_until_cross_acc_1_00_rt_weigth_1_temp_None.pkl.gz"

        knapsack_rmse_until_cross_acc_0_960_with_filter = "./pickle/1f513996_1_until_cross_acc_0_96_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_0_970_with_filter = "./pickle/1f513996_1_until_cross_acc_0_97_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_0_980_with_filter = "./pickle/1f513996_1_until_cross_acc_0_98_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_0_990_with_filter = "./pickle/1f513996_1_until_cross_acc_0_99_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_0_965_with_filter = "./pickle/1f513996_1_until_cross_acc_0_965_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_0_975_with_filter = "./pickle/1f513996_1_until_cross_acc_0_975_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_0_985_with_filter = "./pickle/1f513996_1_until_cross_acc_0_985_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_0_995_with_filter = "./pickle/1f513996_1_until_cross_acc_0_995_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_1_000_with_filter = "./pickle/1f513996_1_until_cross_acc_1_00_rt_weigth_1_temp_None.pkl.gz"

        knapsack_cross_acc_until_cross_acc_0_960_with_filter = "./pickle/5cfd8084_2_until_cross_acc_0_96_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_0_970_with_filter = "./pickle/5cfd8084_2_until_cross_acc_0_97_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_0_980_with_filter = "./pickle/5cfd8084_2_until_cross_acc_0_98_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_0_990_with_filter = "./pickle/5cfd8084_2_until_cross_acc_0_99_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_0_965_with_filter = "./pickle/5cfd8084_2_until_cross_acc_0_965_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_0_975_with_filter = "./pickle/5cfd8084_2_until_cross_acc_0_975_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_0_985_with_filter = "./pickle/5cfd8084_2_until_cross_acc_0_985_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_0_995_with_filter = "./pickle/5cfd8084_2_until_cross_acc_0_995_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_1_000_with_filter = "./pickle/5cfd8084_2_until_cross_acc_1_00_rt_weigth_1_temp_None.pkl.gz"

        knapsack_cross_acc_until_cross_acc_0_960_no_filter = "./pickle/5cfd8084_3_until_cross_acc_0_96_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_0_970_no_filter = "./pickle/5cfd8084_3_until_cross_acc_0_97_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_0_980_no_filter = "./pickle/5cfd8084_3_until_cross_acc_0_98_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_0_990_no_filter = "./pickle/5cfd8084_3_until_cross_acc_0_99_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_0_965_no_filter = "./pickle/5cfd8084_3_until_cross_acc_0_965_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_0_975_no_filter = "./pickle/5cfd8084_3_until_cross_acc_0_975_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_0_985_no_filter = "./pickle/5cfd8084_3_until_cross_acc_0_985_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_0_995_no_filter = "./pickle/5cfd8084_3_until_cross_acc_0_995_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_1_000_no_filter = "./pickle/5cfd8084_3_until_cross_acc_1_00_rt_weigth_1_temp_None.pkl.gz"

        ########## bottom-up: greedy select best idx #################
        greedy_cross_acc_until_stab_0_9800_with_filter = "./pickle/409696c_0_until_stab_0_9800_rt_weigth_1_temp_None.pkl.gz"
        greedy_cross_acc_until_stab_0_9825_with_filter = "./pickle/409696c_0_until_stab_0_9825_rt_weigth_1_temp_None.pkl.gz"
        greedy_cross_acc_until_stab_0_9850_with_filter = "./pickle/409696c_0_until_stab_0_9850_rt_weigth_1_temp_None.pkl.gz"
        greedy_cross_acc_until_stab_0_9875_with_filter = "./pickle/409696c_0_until_stab_0_9875_rt_weigth_1_temp_None.pkl.gz"
        greedy_cross_acc_until_stab_0_9900_with_filter = "./pickle/409696c_0_until_stab_0_9900_rt_weigth_1_temp_None.pkl.gz"
        greedy_cross_acc_until_stab_0_9925_with_filter = "./pickle/409696c_0_until_stab_0_9925_rt_weigth_1_temp_None.pkl.gz"
        greedy_cross_acc_until_stab_0_9950_with_filter = "./pickle/409696c_0_until_stab_0_9950_rt_weigth_1_temp_None.pkl.gz"
        greedy_cross_acc_until_stab_0_9975_with_filter = "./pickle/409696c_0_until_stab_0_9975_rt_weigth_1_temp_None.pkl.gz"

        greedy_cross_acc_until_stab_0_9800_no_filter = "./pickle/409696c_1_until_stab_0_9800_rt_weigth_1_temp_None.pkl.gz"
        greedy_cross_acc_until_stab_0_9825_no_filter = "./pickle/409696c_1_until_stab_0_9825_rt_weigth_1_temp_None.pkl.gz"
        greedy_cross_acc_until_stab_0_9850_no_filter = "./pickle/409696c_1_until_stab_0_9850_rt_weigth_1_temp_None.pkl.gz"
        greedy_cross_acc_until_stab_0_9875_no_filter = "./pickle/409696c_1_until_stab_0_9875_rt_weigth_1_temp_None.pkl.gz"
        greedy_cross_acc_until_stab_0_9900_no_filter = "./pickle/409696c_1_until_stab_0_9900_rt_weigth_1_temp_None.pkl.gz"
        greedy_cross_acc_until_stab_0_9925_no_filter = "./pickle/409696c_1_until_stab_0_9925_rt_weigth_1_temp_None.pkl.gz"
        greedy_cross_acc_until_stab_0_9950_no_filter = "./pickle/409696c_1_until_stab_0_9950_rt_weigth_1_temp_None.pkl.gz"
        greedy_cross_acc_until_stab_0_9975_no_filter = "./pickle/409696c_1_until_stab_0_9975_rt_weigth_1_temp_None.pkl.gz"

        greedy_rmse_until_stab_0_9800_with_filter = "./pickle/409696c_2_until_stab_0_9800_rt_weigth_1_temp_None.pkl.gz"
        greedy_rmse_until_stab_0_9825_with_filter = "./pickle/409696c_2_until_stab_0_9825_rt_weigth_1_temp_None.pkl.gz"
        greedy_rmse_until_stab_0_9850_with_filter = "./pickle/409696c_2_until_stab_0_9850_rt_weigth_1_temp_None.pkl.gz"
        greedy_rmse_until_stab_0_9875_with_filter = "./pickle/409696c_2_until_stab_0_9875_rt_weigth_1_temp_None.pkl.gz"
        greedy_rmse_until_stab_0_9900_with_filter = "./pickle/409696c_2_until_stab_0_9900_rt_weigth_1_temp_None.pkl.gz"
        greedy_rmse_until_stab_0_9925_with_filter = "./pickle/409696c_2_until_stab_0_9925_rt_weigth_1_temp_None.pkl.gz"
        greedy_rmse_until_stab_0_9950_with_filter = "./pickle/409696c_2_until_stab_0_9950_rt_weigth_1_temp_None.pkl.gz"
        greedy_rmse_until_stab_0_9975_with_filter = "./pickle/409696c_2_until_stab_0_9975_rt_weigth_1_temp_None.pkl.gz"

        greedy_rmse_until_stab_0_9800_no_filter = "./pickle/409696c_3_until_stab_0_9800_rt_weigth_1_temp_None.pkl.gz"
        greedy_rmse_until_stab_0_9825_no_filter = "./pickle/409696c_3_until_stab_0_9825_rt_weigth_1_temp_None.pkl.gz"
        greedy_rmse_until_stab_0_9850_no_filter = "./pickle/409696c_3_until_stab_0_9850_rt_weigth_1_temp_None.pkl.gz"
        greedy_rmse_until_stab_0_9875_no_filter = "./pickle/409696c_3_until_stab_0_9875_rt_weigth_1_temp_None.pkl.gz"
        greedy_rmse_until_stab_0_9900_no_filter = "./pickle/409696c_3_until_stab_0_9900_rt_weigth_1_temp_None.pkl.gz"
        greedy_rmse_until_stab_0_9925_no_filter = "./pickle/409696c_3_until_stab_0_9925_rt_weigth_1_temp_None.pkl.gz"
        greedy_rmse_until_stab_0_9950_no_filter = "./pickle/409696c_3_until_stab_0_9950_rt_weigth_1_temp_None.pkl.gz"
        greedy_rmse_until_stab_0_9975_no_filter = "./pickle/409696c_3_until_stab_0_9975_rt_weigth_1_temp_None.pkl.gz"

        ########## bottom-up: greedy top k #################
        greedy_rmse_until_stab_0_9800_top_3_with_filter      = None # wrong config"./pickle/8be74ab_0_until_stab_0_9800_rt_weigth_1_temp_None.pkl.gz"
        greedy_rmse_until_stab_0_9825_top_3_with_filter      = None # wrong config"./pickle/8be74ab_0_until_stab_0_9825_rt_weigth_1_temp_None.pkl.gz"
        greedy_rmse_until_stab_0_9850_top_3_with_filter      = None # wrong config"./pickle/8be74ab_0_until_stab_0_9850_rt_weigth_1_temp_None.pkl.gz"
        greedy_rmse_until_stab_0_9875_top_3_with_filter      = None # wrong config"./pickle/8be74ab_0_until_stab_0_9875_rt_weigth_1_temp_None.pkl.gz"
        greedy_rmse_until_stab_0_9900_top_3_with_filter      = None # wrong config"./pickle/8be74ab_0_until_stab_0_9900_rt_weigth_1_temp_None.pkl.gz"
        greedy_rmse_until_stab_0_9925_top_3_with_filter      = None # wrong config"./pickle/8be74ab_0_until_stab_0_9925_rt_weigth_1_temp_None.pkl.gz"
        greedy_rmse_until_stab_0_9950_top_3_with_filter      = None # wrong config"./pickle/8be74ab_0_until_stab_0_9950_rt_weigth_1_temp_None.pkl.gz"
        greedy_rmse_until_stab_0_9975_top_3_with_filter      = None # wrong config"./pickle/8be74ab_0_until_stab_0_9975_rt_weigth_1_temp_None.pkl.gz"
        greedy_rmse_until_stab_0_9800_top_3_no_filter        = "./pickle/8be74ab_1_until_stab_0_9800_rt_weigth_1_temp_None.pkl.gz"
        greedy_rmse_until_stab_0_9825_top_3_no_filter        = "./pickle/8be74ab_1_until_stab_0_9825_rt_weigth_1_temp_None.pkl.gz"
        greedy_rmse_until_stab_0_9850_top_3_no_filter        = "./pickle/8be74ab_1_until_stab_0_9850_rt_weigth_1_temp_None.pkl.gz"
        greedy_rmse_until_stab_0_9875_top_3_no_filter        = "./pickle/8be74ab_1_until_stab_0_9875_rt_weigth_1_temp_None.pkl.gz"
        greedy_rmse_until_stab_0_9900_top_3_no_filter        = "./pickle/8be74ab_1_until_stab_0_9900_rt_weigth_1_temp_None.pkl.gz"
        greedy_rmse_until_stab_0_9925_top_3_no_filter        = "./pickle/8be74ab_1_until_stab_0_9925_rt_weigth_1_temp_None.pkl.gz"
        greedy_rmse_until_stab_0_9950_top_3_no_filter        = "./pickle/8be74ab_1_until_stab_0_9950_rt_weigth_1_temp_None.pkl.gz"
        greedy_rmse_until_stab_0_9975_top_3_no_filter        = "./pickle/8be74ab_1_until_stab_0_9975_rt_weigth_1_temp_None.pkl.gz"
        greedy_cross_acc_until_stab_0_9800_top_3_with_filter = "./pickle/8be74ab_2_until_stab_0_9800_rt_weigth_1_temp_None.pkl.gz"
        greedy_cross_acc_until_stab_0_9825_top_3_with_filter = "./pickle/8be74ab_2_until_stab_0_9825_rt_weigth_1_temp_None.pkl.gz"
        greedy_cross_acc_until_stab_0_9850_top_3_with_filter = "./pickle/8be74ab_2_until_stab_0_9850_rt_weigth_1_temp_None.pkl.gz"
        greedy_cross_acc_until_stab_0_9875_top_3_with_filter = "./pickle/8be74ab_2_until_stab_0_9875_rt_weigth_1_temp_None.pkl.gz"
        greedy_cross_acc_until_stab_0_9900_top_3_with_filter = "./pickle/8be74ab_2_until_stab_0_9900_rt_weigth_1_temp_None.pkl.gz"
        greedy_cross_acc_until_stab_0_9925_top_3_with_filter = "./pickle/8be74ab_2_until_stab_0_9925_rt_weigth_1_temp_None.pkl.gz"
        greedy_cross_acc_until_stab_0_9950_top_3_with_filter = "./pickle/8be74ab_2_until_stab_0_9950_rt_weigth_1_temp_None.pkl.gz"
        greedy_cross_acc_until_stab_0_9975_top_3_with_filter = "./pickle/8be74ab_2_until_stab_0_9975_rt_weigth_1_temp_None.pkl.gz"
        greedy_cross_acc_until_stab_0_9800_top_3_no_filter   = "./pickle/8be74ab_3_until_stab_0_9800_rt_weigth_1_temp_None.pkl.gz"
        greedy_cross_acc_until_stab_0_9825_top_3_no_filter   = "./pickle/8be74ab_3_until_stab_0_9825_rt_weigth_1_temp_None.pkl.gz"
        greedy_cross_acc_until_stab_0_9850_top_3_no_filter   = "./pickle/8be74ab_3_until_stab_0_9850_rt_weigth_1_temp_None.pkl.gz"
        greedy_cross_acc_until_stab_0_9875_top_3_no_filter   = "./pickle/8be74ab_3_until_stab_0_9875_rt_weigth_1_temp_None.pkl.gz"
        greedy_cross_acc_until_stab_0_9900_top_3_no_filter   = "./pickle/8be74ab_3_until_stab_0_9900_rt_weigth_1_temp_None.pkl.gz"
        greedy_cross_acc_until_stab_0_9925_top_3_no_filter   = "./pickle/8be74ab_3_until_stab_0_9925_rt_weigth_1_temp_None.pkl.gz"
        greedy_cross_acc_until_stab_0_9950_top_3_no_filter   = "./pickle/8be74ab_3_until_stab_0_9950_rt_weigth_1_temp_None.pkl.gz"
        greedy_cross_acc_until_stab_0_9975_top_3_no_filter   = "./pickle/8be74ab_3_until_stab_0_9975_rt_weigth_1_temp_None.pkl.gz"
        greedy_rmse_until_stab_0_9800_top_9_with_filter      = "./pickle/8be74ab_4_until_stab_0_9800_rt_weigth_1_temp_None.pkl.gz"
        greedy_rmse_until_stab_0_9825_top_9_with_filter      = "./pickle/8be74ab_4_until_stab_0_9825_rt_weigth_1_temp_None.pkl.gz"
        greedy_rmse_until_stab_0_9850_top_9_with_filter      = "./pickle/8be74ab_4_until_stab_0_9850_rt_weigth_1_temp_None.pkl.gz"
        greedy_rmse_until_stab_0_9875_top_9_with_filter      = "./pickle/8be74ab_4_until_stab_0_9875_rt_weigth_1_temp_None.pkl.gz"
        greedy_rmse_until_stab_0_9900_top_9_with_filter      = "./pickle/8be74ab_4_until_stab_0_9900_rt_weigth_1_temp_None.pkl.gz"
        greedy_rmse_until_stab_0_9925_top_9_with_filter      = "./pickle/8be74ab_4_until_stab_0_9925_rt_weigth_1_temp_None.pkl.gz"
        greedy_rmse_until_stab_0_9950_top_9_with_filter      = "./pickle/8be74ab_4_until_stab_0_9950_rt_weigth_1_temp_None.pkl.gz"
        greedy_rmse_until_stab_0_9975_top_9_with_filter      = "./pickle/8be74ab_4_until_stab_0_9975_rt_weigth_1_temp_None.pkl.gz"
        greedy_rmse_until_stab_0_9800_top_9_no_filter        = "./pickle/8be74ab_5_until_stab_0_9800_rt_weigth_1_temp_None.pkl.gz"
        greedy_rmse_until_stab_0_9825_top_9_no_filter        = "./pickle/8be74ab_5_until_stab_0_9825_rt_weigth_1_temp_None.pkl.gz"
        greedy_rmse_until_stab_0_9850_top_9_no_filter        = "./pickle/8be74ab_5_until_stab_0_9850_rt_weigth_1_temp_None.pkl.gz"
        greedy_rmse_until_stab_0_9875_top_9_no_filter        = "./pickle/8be74ab_5_until_stab_0_9875_rt_weigth_1_temp_None.pkl.gz"
        greedy_rmse_until_stab_0_9900_top_9_no_filter        = "./pickle/8be74ab_5_until_stab_0_9900_rt_weigth_1_temp_None.pkl.gz"
        greedy_rmse_until_stab_0_9925_top_9_no_filter        = "./pickle/8be74ab_5_until_stab_0_9925_rt_weigth_1_temp_None.pkl.gz"
        greedy_rmse_until_stab_0_9950_top_9_no_filter        = "./pickle/8be74ab_5_until_stab_0_9950_rt_weigth_1_temp_None.pkl.gz"
        greedy_rmse_until_stab_0_9975_top_9_no_filter        = "./pickle/8be74ab_5_until_stab_0_9975_rt_weigth_1_temp_None.pkl.gz"
        greedy_cross_acc_until_stab_0_9800_top_9_with_filter = "./pickle/8be74ab_6_until_stab_0_9800_rt_weigth_1_temp_None.pkl.gz"
        greedy_cross_acc_until_stab_0_9825_top_9_with_filter = "./pickle/8be74ab_6_until_stab_0_9825_rt_weigth_1_temp_None.pkl.gz"
        greedy_cross_acc_until_stab_0_9850_top_9_with_filter = "./pickle/8be74ab_6_until_stab_0_9850_rt_weigth_1_temp_None.pkl.gz"
        greedy_cross_acc_until_stab_0_9875_top_9_with_filter = "./pickle/8be74ab_6_until_stab_0_9875_rt_weigth_1_temp_None.pkl.gz"
        greedy_cross_acc_until_stab_0_9900_top_9_with_filter = "./pickle/8be74ab_6_until_stab_0_9900_rt_weigth_1_temp_None.pkl.gz"
        greedy_cross_acc_until_stab_0_9925_top_9_with_filter = "./pickle/8be74ab_6_until_stab_0_9925_rt_weigth_1_temp_None.pkl.gz"
        greedy_cross_acc_until_stab_0_9950_top_9_with_filter = "./pickle/8be74ab_6_until_stab_0_9950_rt_weigth_1_temp_None.pkl.gz"
        greedy_cross_acc_until_stab_0_9975_top_9_with_filter = "./pickle/8be74ab_6_until_stab_0_9975_rt_weigth_1_temp_None.pkl.gz"
        greedy_cross_acc_until_stab_0_9800_top_9_no_filter   = "./pickle/8be74ab_7_until_stab_0_9800_rt_weigth_1_temp_None.pkl.gz"
        greedy_cross_acc_until_stab_0_9825_top_9_no_filter   = "./pickle/8be74ab_7_until_stab_0_9825_rt_weigth_1_temp_None.pkl.gz"
        greedy_cross_acc_until_stab_0_9850_top_9_no_filter   = "./pickle/8be74ab_7_until_stab_0_9850_rt_weigth_1_temp_None.pkl.gz"
        greedy_cross_acc_until_stab_0_9875_top_9_no_filter   = "./pickle/8be74ab_7_until_stab_0_9875_rt_weigth_1_temp_None.pkl.gz"
        greedy_cross_acc_until_stab_0_9900_top_9_no_filter   = "./pickle/8be74ab_7_until_stab_0_9900_rt_weigth_1_temp_None.pkl.gz"
        greedy_cross_acc_until_stab_0_9925_top_9_no_filter   = "./pickle/8be74ab_7_until_stab_0_9925_rt_weigth_1_temp_None.pkl.gz"
        greedy_cross_acc_until_stab_0_9950_top_9_no_filter   = "./pickle/8be74ab_7_until_stab_0_9950_rt_weigth_1_temp_None.pkl.gz"
        greedy_cross_acc_until_stab_0_9975_top_9_no_filter   = "./pickle/8be74ab_7_until_stab_0_9975_rt_weigth_1_temp_None.pkl.gz"
        greedy_rmse_until_stab_0_9800_top_27_with_filter      = "./pickle/8be74ab_8_until_stab_0_9800_rt_weigth_1_temp_None.pkl.gz"
        greedy_rmse_until_stab_0_9825_top_27_with_filter      = "./pickle/8be74ab_8_until_stab_0_9825_rt_weigth_1_temp_None.pkl.gz"
        greedy_rmse_until_stab_0_9850_top_27_with_filter      = "./pickle/8be74ab_8_until_stab_0_9850_rt_weigth_1_temp_None.pkl.gz"
        greedy_rmse_until_stab_0_9875_top_27_with_filter      = "./pickle/8be74ab_8_until_stab_0_9875_rt_weigth_1_temp_None.pkl.gz"
        greedy_rmse_until_stab_0_9900_top_27_with_filter      = "./pickle/8be74ab_8_until_stab_0_9900_rt_weigth_1_temp_None.pkl.gz"
        greedy_rmse_until_stab_0_9925_top_27_with_filter      = "./pickle/8be74ab_8_until_stab_0_9925_rt_weigth_1_temp_None.pkl.gz"
        greedy_rmse_until_stab_0_9950_top_27_with_filter      = "./pickle/8be74ab_8_until_stab_0_9950_rt_weigth_1_temp_None.pkl.gz"
        greedy_rmse_until_stab_0_9975_top_27_with_filter      = "./pickle/8be74ab_8_until_stab_0_9975_rt_weigth_1_temp_None.pkl.gz"
        greedy_rmse_until_stab_0_9800_top_27_no_filter        = "./pickle/8be74ab_9_until_stab_0_9800_rt_weigth_1_temp_None.pkl.gz"
        greedy_rmse_until_stab_0_9825_top_27_no_filter        = "./pickle/8be74ab_9_until_stab_0_9825_rt_weigth_1_temp_None.pkl.gz"
        greedy_rmse_until_stab_0_9850_top_27_no_filter        = "./pickle/8be74ab_9_until_stab_0_9850_rt_weigth_1_temp_None.pkl.gz"
        greedy_rmse_until_stab_0_9875_top_27_no_filter        = "./pickle/8be74ab_9_until_stab_0_9875_rt_weigth_1_temp_None.pkl.gz"
        greedy_rmse_until_stab_0_9900_top_27_no_filter        = "./pickle/8be74ab_9_until_stab_0_9900_rt_weigth_1_temp_None.pkl.gz"
        greedy_rmse_until_stab_0_9925_top_27_no_filter        = "./pickle/8be74ab_9_until_stab_0_9925_rt_weigth_1_temp_None.pkl.gz"
        greedy_rmse_until_stab_0_9950_top_27_no_filter        = "./pickle/8be74ab_9_until_stab_0_9950_rt_weigth_1_temp_None.pkl.gz"
        greedy_rmse_until_stab_0_9975_top_27_no_filter        = "./pickle/8be74ab_9_until_stab_0_9975_rt_weigth_1_temp_None.pkl.gz"
        greedy_cross_acc_until_stab_0_9800_top_27_with_filter = "./pickle/8be74ab_10_until_stab_0_9800_rt_weigth_1_temp_None.pkl.gz"
        greedy_cross_acc_until_stab_0_9825_top_27_with_filter = "./pickle/8be74ab_10_until_stab_0_9825_rt_weigth_1_temp_None.pkl.gz"
        greedy_cross_acc_until_stab_0_9850_top_27_with_filter = "./pickle/8be74ab_10_until_stab_0_9850_rt_weigth_1_temp_None.pkl.gz"
        greedy_cross_acc_until_stab_0_9875_top_27_with_filter = "./pickle/8be74ab_10_until_stab_0_9875_rt_weigth_1_temp_None.pkl.gz"
        greedy_cross_acc_until_stab_0_9900_top_27_with_filter = "./pickle/8be74ab_10_until_stab_0_9900_rt_weigth_1_temp_None.pkl.gz"
        greedy_cross_acc_until_stab_0_9925_top_27_with_filter = "./pickle/8be74ab_10_until_stab_0_9925_rt_weigth_1_temp_None.pkl.gz"
        greedy_cross_acc_until_stab_0_9950_top_27_with_filter = "./pickle/8be74ab_10_until_stab_0_9950_rt_weigth_1_temp_None.pkl.gz"
        greedy_cross_acc_until_stab_0_9975_top_27_with_filter = "./pickle/8be74ab_10_until_stab_0_9975_rt_weigth_1_temp_None.pkl.gz"
        greedy_cross_acc_until_stab_0_9800_top_27_no_filter   = "./pickle/8be74ab_11_until_stab_0_9800_rt_weigth_1_temp_None.pkl.gz"
        greedy_cross_acc_until_stab_0_9825_top_27_no_filter   = "./pickle/8be74ab_11_until_stab_0_9825_rt_weigth_1_temp_None.pkl.gz"
        greedy_cross_acc_until_stab_0_9850_top_27_no_filter   = "./pickle/8be74ab_11_until_stab_0_9850_rt_weigth_1_temp_None.pkl.gz"
        greedy_cross_acc_until_stab_0_9875_top_27_no_filter   = "./pickle/8be74ab_11_until_stab_0_9875_rt_weigth_1_temp_None.pkl.gz"
        greedy_cross_acc_until_stab_0_9900_top_27_no_filter   = "./pickle/8be74ab_11_until_stab_0_9900_rt_weigth_1_temp_None.pkl.gz"
        greedy_cross_acc_until_stab_0_9925_top_27_no_filter   = "./pickle/8be74ab_11_until_stab_0_9925_rt_weigth_1_temp_None.pkl.gz"
        greedy_cross_acc_until_stab_0_9950_top_27_no_filter   = "./pickle/8be74ab_11_until_stab_0_9950_rt_weigth_1_temp_None.pkl.gz"
        greedy_cross_acc_until_stab_0_9975_top_27_no_filter   = "./pickle/8be74ab_11_until_stab_0_9975_rt_weigth_1_temp_None.pkl.gz"

        ########## bottom-up: knapsack top k #################
        knapsack_rmse_until_cross_acc_0_960_top_3_with_filter      = "./pickle/8ce47871_0_until_cross_acc_0_96_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_0_970_top_3_with_filter      = "./pickle/8ce47871_0_until_cross_acc_0_97_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_0_980_top_3_with_filter      = "./pickle/8ce47871_0_until_cross_acc_0_98_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_0_990_top_3_with_filter      = "./pickle/8ce47871_0_until_cross_acc_0_99_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_0_965_top_3_with_filter      = "./pickle/8ce47871_0_until_cross_acc_0_965_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_0_975_top_3_with_filter      = "./pickle/8ce47871_0_until_cross_acc_0_975_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_0_985_top_3_with_filter      = "./pickle/8ce47871_0_until_cross_acc_0_985_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_0_995_top_3_with_filter      = "./pickle/8ce47871_0_until_cross_acc_0_995_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_1_000_top_3_with_filter      = "./pickle/8ce47871_0_until_cross_acc_1_00_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_0_960_top_3_no_filter        = "./pickle/8ce47871_1_until_cross_acc_0_96_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_0_970_top_3_no_filter        = "./pickle/8ce47871_1_until_cross_acc_0_97_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_0_980_top_3_no_filter        = "./pickle/8ce47871_1_until_cross_acc_0_98_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_0_990_top_3_no_filter        = "./pickle/8ce47871_1_until_cross_acc_0_99_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_0_965_top_3_no_filter        = "./pickle/8ce47871_1_until_cross_acc_0_965_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_0_975_top_3_no_filter        = "./pickle/8ce47871_1_until_cross_acc_0_975_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_0_985_top_3_no_filter        = "./pickle/8ce47871_1_until_cross_acc_0_985_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_0_995_top_3_no_filter        = "./pickle/8ce47871_1_until_cross_acc_0_995_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_1_000_top_3_no_filter        = "./pickle/8ce47871_1_until_cross_acc_1_00_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_0_960_top_3_with_filter = "./pickle/8ce47871_2_until_cross_acc_0_96_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_0_970_top_3_with_filter = "./pickle/8ce47871_2_until_cross_acc_0_97_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_0_980_top_3_with_filter = "./pickle/8ce47871_2_until_cross_acc_0_98_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_0_990_top_3_with_filter = "./pickle/8ce47871_2_until_cross_acc_0_99_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_0_965_top_3_with_filter = "./pickle/8ce47871_2_until_cross_acc_0_965_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_0_975_top_3_with_filter = "./pickle/8ce47871_2_until_cross_acc_0_975_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_0_985_top_3_with_filter = "./pickle/8ce47871_2_until_cross_acc_0_985_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_0_995_top_3_with_filter = "./pickle/8ce47871_2_until_cross_acc_0_995_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_1_000_top_3_with_filter = "./pickle/8ce47871_2_until_cross_acc_1_00_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_0_960_top_3_no_filter   = "./pickle/bf04035c_3_until_cross_acc_0_96_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_0_970_top_3_no_filter   = "./pickle/bf04035c_3_until_cross_acc_0_97_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_0_980_top_3_no_filter   = "./pickle/bf04035c_3_until_cross_acc_0_98_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_0_990_top_3_no_filter   = "./pickle/bf04035c_3_until_cross_acc_0_99_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_0_965_top_3_no_filter   = "./pickle/bf04035c_3_until_cross_acc_0_965_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_0_975_top_3_no_filter   = "./pickle/bf04035c_3_until_cross_acc_0_975_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_0_985_top_3_no_filter   = "./pickle/bf04035c_3_until_cross_acc_0_985_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_0_995_top_3_no_filter   = "./pickle/bf04035c_3_until_cross_acc_0_995_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_1_000_top_3_no_filter   = "./pickle/bf04035c_3_until_cross_acc_1_00_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_0_960_top_9_with_filter      = "./pickle/8be74aba_4_until_cross_acc_0_96_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_0_970_top_9_with_filter      = "./pickle/8be74aba_4_until_cross_acc_0_97_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_0_980_top_9_with_filter      = "./pickle/8be74aba_4_until_cross_acc_0_98_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_0_990_top_9_with_filter      = "./pickle/8be74aba_4_until_cross_acc_0_99_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_0_965_top_9_with_filter      = "./pickle/8be74aba_4_until_cross_acc_0_965_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_0_975_top_9_with_filter      = "./pickle/8be74aba_4_until_cross_acc_0_975_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_0_985_top_9_with_filter      = "./pickle/8be74aba_4_until_cross_acc_0_985_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_0_995_top_9_with_filter      = "./pickle/8be74aba_4_until_cross_acc_0_995_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_1_000_top_9_with_filter      = "./pickle/8be74aba_4_until_cross_acc_1_00_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_0_960_top_9_no_filter        = "./pickle/8be74aba_5_until_cross_acc_0_96_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_0_970_top_9_no_filter        = "./pickle/8be74aba_5_until_cross_acc_0_97_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_0_980_top_9_no_filter        = "./pickle/8be74aba_5_until_cross_acc_0_98_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_0_990_top_9_no_filter        = "./pickle/8be74aba_5_until_cross_acc_0_99_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_0_965_top_9_no_filter        = "./pickle/8be74aba_5_until_cross_acc_0_965_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_0_975_top_9_no_filter        = "./pickle/8be74aba_5_until_cross_acc_0_975_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_0_985_top_9_no_filter        = "./pickle/8be74aba_5_until_cross_acc_0_985_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_0_995_top_9_no_filter        = "./pickle/8be74aba_5_until_cross_acc_0_995_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_1_000_top_9_no_filter        = "./pickle/8be74aba_5_until_cross_acc_1_00_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_0_960_top_9_with_filter = "./pickle/8be74aba_6_until_cross_acc_0_96_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_0_970_top_9_with_filter = "./pickle/8be74aba_6_until_cross_acc_0_97_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_0_980_top_9_with_filter = "./pickle/8be74aba_6_until_cross_acc_0_98_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_0_990_top_9_with_filter = "./pickle/8be74aba_6_until_cross_acc_0_99_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_0_965_top_9_with_filter = "./pickle/8be74aba_6_until_cross_acc_0_965_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_0_975_top_9_with_filter = "./pickle/8be74aba_6_until_cross_acc_0_975_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_0_985_top_9_with_filter = "./pickle/8be74aba_6_until_cross_acc_0_985_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_0_995_top_9_with_filter = "./pickle/8be74aba_6_until_cross_acc_0_995_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_1_000_top_9_with_filter = "./pickle/8be74aba_6_until_cross_acc_1_00_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_0_960_top_9_no_filter   = "./pickle/8be74aba_7_until_cross_acc_0_96_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_0_970_top_9_no_filter   = "./pickle/8be74aba_7_until_cross_acc_0_97_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_0_980_top_9_no_filter   = "./pickle/8be74aba_7_until_cross_acc_0_98_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_0_990_top_9_no_filter   = "./pickle/8be74aba_7_until_cross_acc_0_99_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_0_965_top_9_no_filter   = "./pickle/8be74aba_7_until_cross_acc_0_965_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_0_975_top_9_no_filter   = "./pickle/8be74aba_7_until_cross_acc_0_975_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_0_985_top_9_no_filter   = "./pickle/8be74aba_7_until_cross_acc_0_985_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_0_995_top_9_no_filter   = "./pickle/8be74aba_7_until_cross_acc_0_995_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_1_000_top_9_no_filter   = "./pickle/8be74aba_7_until_cross_acc_1_00_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_0_960_top_27_with_filter      = "./pickle/8be74aba_8_until_cross_acc_0_96_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_0_970_top_27_with_filter      = "./pickle/8be74aba_8_until_cross_acc_0_97_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_0_980_top_27_with_filter      = "./pickle/8be74aba_8_until_cross_acc_0_98_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_0_990_top_27_with_filter      = "./pickle/8be74aba_8_until_cross_acc_0_99_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_0_965_top_27_with_filter      = "./pickle/8be74aba_8_until_cross_acc_0_965_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_0_975_top_27_with_filter      = "./pickle/8be74aba_8_until_cross_acc_0_975_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_0_985_top_27_with_filter      = "./pickle/8be74aba_8_until_cross_acc_0_985_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_0_995_top_27_with_filter      = "./pickle/8be74aba_8_until_cross_acc_0_995_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_1_000_top_27_with_filter      = "./pickle/8be74aba_8_until_cross_acc_1_00_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_0_960_top_27_no_filter        = "./pickle/8be74aba_9_until_cross_acc_0_96_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_0_970_top_27_no_filter        = "./pickle/8be74aba_9_until_cross_acc_0_97_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_0_980_top_27_no_filter        = "./pickle/8be74aba_9_until_cross_acc_0_98_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_0_990_top_27_no_filter        = "./pickle/8be74aba_9_until_cross_acc_0_99_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_0_965_top_27_no_filter        = "./pickle/8be74aba_9_until_cross_acc_0_965_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_0_975_top_27_no_filter        = "./pickle/8be74aba_9_until_cross_acc_0_975_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_0_985_top_27_no_filter        = "./pickle/8be74aba_9_until_cross_acc_0_985_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_0_995_top_27_no_filter        = "./pickle/8be74aba_9_until_cross_acc_0_995_rt_weigth_1_temp_None.pkl.gz"
        knapsack_rmse_until_cross_acc_1_000_top_27_no_filter        = "./pickle/8be74aba_9_until_cross_acc_1_00_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_0_960_top_27_with_filter = "./pickle/8be74aba_10_until_cross_acc_0_96_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_0_970_top_27_with_filter = "./pickle/8be74aba_10_until_cross_acc_0_97_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_0_980_top_27_with_filter = "./pickle/8be74aba_10_until_cross_acc_0_98_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_0_990_top_27_with_filter = "./pickle/8be74aba_10_until_cross_acc_0_99_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_0_965_top_27_with_filter = "./pickle/8be74aba_10_until_cross_acc_0_965_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_0_975_top_27_with_filter = "./pickle/8be74aba_10_until_cross_acc_0_975_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_0_985_top_27_with_filter = "./pickle/8be74aba_10_until_cross_acc_0_985_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_0_995_top_27_with_filter = "./pickle/8be74aba_10_until_cross_acc_0_995_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_1_000_top_27_with_filter = "./pickle/8be74aba_10_until_cross_acc_1_00_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_0_960_top_27_no_filter   = "./pickle/8be74aba_11_until_cross_acc_0_96_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_0_970_top_27_no_filter   = "./pickle/8be74aba_11_until_cross_acc_0_97_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_0_980_top_27_no_filter   = "./pickle/8be74aba_11_until_cross_acc_0_98_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_0_990_top_27_no_filter   = "./pickle/8be74aba_11_until_cross_acc_0_99_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_0_965_top_27_no_filter   = "./pickle/8be74aba_11_until_cross_acc_0_965_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_0_975_top_27_no_filter   = "./pickle/8be74aba_11_until_cross_acc_0_975_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_0_985_top_27_no_filter   = "./pickle/8be74aba_11_until_cross_acc_0_985_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_0_995_top_27_no_filter   = "./pickle/8be74aba_11_until_cross_acc_0_995_rt_weigth_1_temp_None.pkl.gz"
        knapsack_cross_acc_until_cross_acc_1_000_top_27_no_filter   = "./pickle/8be74aba_11_until_cross_acc_1_00_rt_weigth_1_temp_None.pkl.gz"

        ########### RANDOM TIMEOUT ###############
        random_timeout_with_filter = "./pickle/955dd53_0_not_needed_rt_weigth_1_temp_None.pkl.gz"
        random_timeout_no_filter = "./pickle/955dd53_1_not_needed_rt_weigth_1_temp_None.pkl.gz"

        al_low_delta_rt = [0.0541, 0.1035]
        al_high_delta_acc = [0.9048, 0.9233]

        plt.figure(figsize=(10, 6))


        # "pareto", "compare estimator"
        #self.get_all_measurements_instance_wise([delta_0_4, delta_0_5, delta_0_6, delta_0_7, delta_0_8, delta_0_9, delta_0_4_no_filter, delta_0_5_no_filter, delta_0_6_no_filter, delta_0_7_no_filter, delta_0_8_no_filter, delta_0_9_no_filter], "pareto")#'compare', "$\mathrm{rand}$")

        #self.get_all_measurements_random_timeout([random_timeout_with_filter, random_timeout_no_filter], 'compare', "$\mathrm{rand}$")

        self.get_all_measurements_bottom_up([knapsack_rmse_until_cross_acc_0_960_no_filter               ,knapsack_rmse_until_cross_acc_0_970_no_filter               ,knapsack_rmse_until_cross_acc_0_980_no_filter               ,knapsack_rmse_until_cross_acc_0_990_no_filter               ,knapsack_rmse_until_cross_acc_0_965_no_filter               ,knapsack_rmse_until_cross_acc_0_975_no_filter               ,knapsack_rmse_until_cross_acc_0_985_no_filter               ,knapsack_rmse_until_cross_acc_0_995_no_filter               ,knapsack_rmse_until_cross_acc_1_000_no_filter               ,knapsack_rmse_until_cross_acc_0_960_with_filter             ,knapsack_rmse_until_cross_acc_0_970_with_filter             ,knapsack_rmse_until_cross_acc_0_980_with_filter             ,knapsack_rmse_until_cross_acc_0_990_with_filter             ,knapsack_rmse_until_cross_acc_0_965_with_filter             ,knapsack_rmse_until_cross_acc_0_975_with_filter             ,knapsack_rmse_until_cross_acc_0_985_with_filter             ,knapsack_rmse_until_cross_acc_0_995_with_filter             ,knapsack_rmse_until_cross_acc_1_000_with_filter             ,knapsack_cross_acc_until_cross_acc_0_960_with_filter        ,knapsack_cross_acc_until_cross_acc_0_970_with_filter        ,knapsack_cross_acc_until_cross_acc_0_980_with_filter        ,knapsack_cross_acc_until_cross_acc_0_990_with_filter        ,knapsack_cross_acc_until_cross_acc_0_965_with_filter        ,knapsack_cross_acc_until_cross_acc_0_975_with_filter        ,knapsack_cross_acc_until_cross_acc_0_985_with_filter        ,knapsack_cross_acc_until_cross_acc_0_995_with_filter        ,knapsack_cross_acc_until_cross_acc_1_000_with_filter        ,knapsack_cross_acc_until_cross_acc_0_960_no_filter          ,knapsack_cross_acc_until_cross_acc_0_970_no_filter          ,knapsack_cross_acc_until_cross_acc_0_980_no_filter          ,knapsack_cross_acc_until_cross_acc_0_990_no_filter          ,knapsack_cross_acc_until_cross_acc_0_965_no_filter          ,knapsack_cross_acc_until_cross_acc_0_975_no_filter          ,knapsack_cross_acc_until_cross_acc_0_985_no_filter          ,knapsack_cross_acc_until_cross_acc_0_995_no_filter          ,knapsack_cross_acc_until_cross_acc_1_000_no_filter          ,greedy_cross_acc_until_stab_0_9800_with_filter              ,greedy_cross_acc_until_stab_0_9825_with_filter              ,greedy_cross_acc_until_stab_0_9850_with_filter              ,greedy_cross_acc_until_stab_0_9875_with_filter              ,greedy_cross_acc_until_stab_0_9900_with_filter              ,greedy_cross_acc_until_stab_0_9925_with_filter              ,greedy_cross_acc_until_stab_0_9950_with_filter              ,greedy_cross_acc_until_stab_0_9975_with_filter              ,greedy_cross_acc_until_stab_0_9800_no_filter                ,greedy_cross_acc_until_stab_0_9825_no_filter                ,greedy_cross_acc_until_stab_0_9850_no_filter                ,greedy_cross_acc_until_stab_0_9875_no_filter                ,greedy_cross_acc_until_stab_0_9900_no_filter                ,greedy_cross_acc_until_stab_0_9925_no_filter                ,greedy_cross_acc_until_stab_0_9950_no_filter                ,greedy_cross_acc_until_stab_0_9975_no_filter                ,greedy_rmse_until_stab_0_9800_with_filter                   ,greedy_rmse_until_stab_0_9825_with_filter                   ,greedy_rmse_until_stab_0_9850_with_filter                   ,greedy_rmse_until_stab_0_9875_with_filter                   ,greedy_rmse_until_stab_0_9900_with_filter                   ,greedy_rmse_until_stab_0_9925_with_filter                   ,greedy_rmse_until_stab_0_9950_with_filter                   ,greedy_rmse_until_stab_0_9975_with_filter                   ,greedy_rmse_until_stab_0_9800_no_filter                     ,greedy_rmse_until_stab_0_9825_no_filter                     ,greedy_rmse_until_stab_0_9850_no_filter                     ,greedy_rmse_until_stab_0_9875_no_filter                     ,greedy_rmse_until_stab_0_9900_no_filter                     ,greedy_rmse_until_stab_0_9925_no_filter                     ,greedy_rmse_until_stab_0_9950_no_filter                     ,greedy_rmse_until_stab_0_9975_no_filter                     ,greedy_rmse_until_stab_0_9800_top_3_no_filter               ,greedy_rmse_until_stab_0_9825_top_3_no_filter               ,greedy_rmse_until_stab_0_9850_top_3_no_filter               ,greedy_rmse_until_stab_0_9875_top_3_no_filter               ,greedy_rmse_until_stab_0_9900_top_3_no_filter               ,greedy_rmse_until_stab_0_9925_top_3_no_filter               ,greedy_rmse_until_stab_0_9950_top_3_no_filter               ,greedy_rmse_until_stab_0_9975_top_3_no_filter               ,greedy_cross_acc_until_stab_0_9800_top_3_with_filter        ,greedy_cross_acc_until_stab_0_9825_top_3_with_filter        ,greedy_cross_acc_until_stab_0_9850_top_3_with_filter        ,greedy_cross_acc_until_stab_0_9875_top_3_with_filter        ,greedy_cross_acc_until_stab_0_9900_top_3_with_filter        ,greedy_cross_acc_until_stab_0_9925_top_3_with_filter        ,greedy_cross_acc_until_stab_0_9950_top_3_with_filter        ,greedy_cross_acc_until_stab_0_9975_top_3_with_filter        ,greedy_cross_acc_until_stab_0_9800_top_3_no_filter          ,greedy_cross_acc_until_stab_0_9825_top_3_no_filter          ,greedy_cross_acc_until_stab_0_9850_top_3_no_filter          ,greedy_cross_acc_until_stab_0_9875_top_3_no_filter          ,greedy_cross_acc_until_stab_0_9900_top_3_no_filter          ,greedy_cross_acc_until_stab_0_9925_top_3_no_filter          ,greedy_cross_acc_until_stab_0_9950_top_3_no_filter          ,greedy_cross_acc_until_stab_0_9975_top_3_no_filter          ,greedy_rmse_until_stab_0_9800_top_9_with_filter             ,greedy_rmse_until_stab_0_9825_top_9_with_filter             ,greedy_rmse_until_stab_0_9850_top_9_with_filter             ,greedy_rmse_until_stab_0_9875_top_9_with_filter             ,greedy_rmse_until_stab_0_9900_top_9_with_filter             ,greedy_rmse_until_stab_0_9925_top_9_with_filter             ,greedy_rmse_until_stab_0_9950_top_9_with_filter             ,greedy_rmse_until_stab_0_9975_top_9_with_filter             ,greedy_rmse_until_stab_0_9800_top_9_no_filter               ,greedy_rmse_until_stab_0_9825_top_9_no_filter               ,greedy_rmse_until_stab_0_9850_top_9_no_filter               ,greedy_rmse_until_stab_0_9875_top_9_no_filter               ,greedy_rmse_until_stab_0_9900_top_9_no_filter               ,greedy_rmse_until_stab_0_9925_top_9_no_filter               ,greedy_rmse_until_stab_0_9950_top_9_no_filter               ,greedy_rmse_until_stab_0_9975_top_9_no_filter               ,greedy_cross_acc_until_stab_0_9800_top_9_with_filter        ,greedy_cross_acc_until_stab_0_9825_top_9_with_filter        ,greedy_cross_acc_until_stab_0_9850_top_9_with_filter        ,greedy_cross_acc_until_stab_0_9875_top_9_with_filter        ,greedy_cross_acc_until_stab_0_9900_top_9_with_filter        ,greedy_cross_acc_until_stab_0_9925_top_9_with_filter        ,greedy_cross_acc_until_stab_0_9950_top_9_with_filter        ,greedy_cross_acc_until_stab_0_9975_top_9_with_filter        ,greedy_cross_acc_until_stab_0_9800_top_9_no_filter          ,greedy_cross_acc_until_stab_0_9825_top_9_no_filter          ,greedy_cross_acc_until_stab_0_9850_top_9_no_filter          ,greedy_cross_acc_until_stab_0_9875_top_9_no_filter          ,greedy_cross_acc_until_stab_0_9900_top_9_no_filter          ,greedy_cross_acc_until_stab_0_9925_top_9_no_filter          ,greedy_cross_acc_until_stab_0_9950_top_9_no_filter          ,greedy_cross_acc_until_stab_0_9975_top_9_no_filter          ,greedy_rmse_until_stab_0_9800_top_27_with_filter            ,greedy_rmse_until_stab_0_9825_top_27_with_filter            ,greedy_rmse_until_stab_0_9850_top_27_with_filter            ,greedy_rmse_until_stab_0_9875_top_27_with_filter            ,greedy_rmse_until_stab_0_9900_top_27_with_filter            ,greedy_rmse_until_stab_0_9925_top_27_with_filter            ,greedy_rmse_until_stab_0_9950_top_27_with_filter            ,greedy_rmse_until_stab_0_9975_top_27_with_filter            ,greedy_rmse_until_stab_0_9800_top_27_no_filter              ,greedy_rmse_until_stab_0_9825_top_27_no_filter              ,greedy_rmse_until_stab_0_9850_top_27_no_filter              ,greedy_rmse_until_stab_0_9875_top_27_no_filter              ,greedy_rmse_until_stab_0_9900_top_27_no_filter              ,greedy_rmse_until_stab_0_9925_top_27_no_filter              ,greedy_rmse_until_stab_0_9950_top_27_no_filter              ,greedy_rmse_until_stab_0_9975_top_27_no_filter              ,greedy_cross_acc_until_stab_0_9800_top_27_with_filter       ,greedy_cross_acc_until_stab_0_9825_top_27_with_filter       ,greedy_cross_acc_until_stab_0_9850_top_27_with_filter       ,greedy_cross_acc_until_stab_0_9875_top_27_with_filter       ,greedy_cross_acc_until_stab_0_9900_top_27_with_filter       ,greedy_cross_acc_until_stab_0_9925_top_27_with_filter       ,greedy_cross_acc_until_stab_0_9950_top_27_with_filter       ,greedy_cross_acc_until_stab_0_9975_top_27_with_filter       ,greedy_cross_acc_until_stab_0_9800_top_27_no_filter         ,greedy_cross_acc_until_stab_0_9825_top_27_no_filter         ,greedy_cross_acc_until_stab_0_9850_top_27_no_filter         ,greedy_cross_acc_until_stab_0_9875_top_27_no_filter         ,greedy_cross_acc_until_stab_0_9900_top_27_no_filter         ,greedy_cross_acc_until_stab_0_9925_top_27_no_filter         ,greedy_cross_acc_until_stab_0_9950_top_27_no_filter         ,greedy_cross_acc_until_stab_0_9975_top_27_no_filter         ,knapsack_rmse_until_cross_acc_0_960_top_3_with_filter       ,knapsack_rmse_until_cross_acc_0_970_top_3_with_filter       ,knapsack_rmse_until_cross_acc_0_980_top_3_with_filter       ,knapsack_rmse_until_cross_acc_0_990_top_3_with_filter       ,knapsack_rmse_until_cross_acc_0_965_top_3_with_filter       ,knapsack_rmse_until_cross_acc_0_975_top_3_with_filter       ,knapsack_rmse_until_cross_acc_0_985_top_3_with_filter       ,knapsack_rmse_until_cross_acc_0_995_top_3_with_filter       ,knapsack_rmse_until_cross_acc_1_000_top_3_with_filter       ,knapsack_rmse_until_cross_acc_0_960_top_3_no_filter         ,knapsack_rmse_until_cross_acc_0_970_top_3_no_filter         ,knapsack_rmse_until_cross_acc_0_980_top_3_no_filter         ,knapsack_rmse_until_cross_acc_0_990_top_3_no_filter         ,knapsack_rmse_until_cross_acc_0_965_top_3_no_filter         ,knapsack_rmse_until_cross_acc_0_975_top_3_no_filter         ,knapsack_rmse_until_cross_acc_0_985_top_3_no_filter         ,knapsack_rmse_until_cross_acc_0_995_top_3_no_filter         ,knapsack_rmse_until_cross_acc_1_000_top_3_no_filter         ,knapsack_cross_acc_until_cross_acc_0_960_top_3_with_filter  ,knapsack_cross_acc_until_cross_acc_0_970_top_3_with_filter  ,knapsack_cross_acc_until_cross_acc_0_980_top_3_with_filter  ,knapsack_cross_acc_until_cross_acc_0_990_top_3_with_filter  ,knapsack_cross_acc_until_cross_acc_0_965_top_3_with_filter  ,knapsack_cross_acc_until_cross_acc_0_975_top_3_with_filter  ,knapsack_cross_acc_until_cross_acc_0_985_top_3_with_filter  ,knapsack_cross_acc_until_cross_acc_0_995_top_3_with_filter  ,knapsack_cross_acc_until_cross_acc_1_000_top_3_with_filter  ,knapsack_cross_acc_until_cross_acc_0_960_top_3_no_filter    ,knapsack_cross_acc_until_cross_acc_0_970_top_3_no_filter    ,knapsack_cross_acc_until_cross_acc_0_980_top_3_no_filter    ,knapsack_cross_acc_until_cross_acc_0_990_top_3_no_filter    ,knapsack_cross_acc_until_cross_acc_0_965_top_3_no_filter    ,knapsack_cross_acc_until_cross_acc_0_975_top_3_no_filter    ,knapsack_cross_acc_until_cross_acc_0_985_top_3_no_filter    ,knapsack_cross_acc_until_cross_acc_0_995_top_3_no_filter    ,knapsack_cross_acc_until_cross_acc_1_000_top_3_no_filter    ,knapsack_rmse_until_cross_acc_0_960_top_9_with_filter       ,knapsack_rmse_until_cross_acc_0_970_top_9_with_filter       ,knapsack_rmse_until_cross_acc_0_980_top_9_with_filter       ,knapsack_rmse_until_cross_acc_0_990_top_9_with_filter       ,knapsack_rmse_until_cross_acc_0_965_top_9_with_filter       ,knapsack_rmse_until_cross_acc_0_975_top_9_with_filter       ,knapsack_rmse_until_cross_acc_0_985_top_9_with_filter       ,knapsack_rmse_until_cross_acc_0_995_top_9_with_filter       ,knapsack_rmse_until_cross_acc_1_000_top_9_with_filter       ,knapsack_rmse_until_cross_acc_0_960_top_9_no_filter         ,knapsack_rmse_until_cross_acc_0_970_top_9_no_filter         ,knapsack_rmse_until_cross_acc_0_980_top_9_no_filter         ,knapsack_rmse_until_cross_acc_0_990_top_9_no_filter         ,knapsack_rmse_until_cross_acc_0_965_top_9_no_filter         ,knapsack_rmse_until_cross_acc_0_975_top_9_no_filter         ,knapsack_rmse_until_cross_acc_0_985_top_9_no_filter         ,knapsack_rmse_until_cross_acc_0_995_top_9_no_filter         ,knapsack_rmse_until_cross_acc_1_000_top_9_no_filter         ,knapsack_cross_acc_until_cross_acc_0_960_top_9_with_filter  ,knapsack_cross_acc_until_cross_acc_0_970_top_9_with_filter  ,knapsack_cross_acc_until_cross_acc_0_980_top_9_with_filter  ,knapsack_cross_acc_until_cross_acc_0_990_top_9_with_filter  ,knapsack_cross_acc_until_cross_acc_0_965_top_9_with_filter  ,knapsack_cross_acc_until_cross_acc_0_975_top_9_with_filter  ,knapsack_cross_acc_until_cross_acc_0_985_top_9_with_filter  ,knapsack_cross_acc_until_cross_acc_0_995_top_9_with_filter  ,knapsack_cross_acc_until_cross_acc_1_000_top_9_with_filter  ,knapsack_cross_acc_until_cross_acc_0_960_top_9_no_filter    ,knapsack_cross_acc_until_cross_acc_0_970_top_9_no_filter    ,knapsack_cross_acc_until_cross_acc_0_980_top_9_no_filter    ,knapsack_cross_acc_until_cross_acc_0_990_top_9_no_filter    ,knapsack_cross_acc_until_cross_acc_0_965_top_9_no_filter    ,knapsack_cross_acc_until_cross_acc_0_975_top_9_no_filter    ,knapsack_cross_acc_until_cross_acc_0_985_top_9_no_filter    ,knapsack_cross_acc_until_cross_acc_0_995_top_9_no_filter    ,knapsack_cross_acc_until_cross_acc_1_000_top_9_no_filter    ,knapsack_rmse_until_cross_acc_0_960_top_27_with_filter      ,knapsack_rmse_until_cross_acc_0_970_top_27_with_filter      ,knapsack_rmse_until_cross_acc_0_980_top_27_with_filter      ,knapsack_rmse_until_cross_acc_0_990_top_27_with_filter      ,knapsack_rmse_until_cross_acc_0_965_top_27_with_filter      ,knapsack_rmse_until_cross_acc_0_975_top_27_with_filter      ,knapsack_rmse_until_cross_acc_0_985_top_27_with_filter      ,knapsack_rmse_until_cross_acc_0_995_top_27_with_filter      ,knapsack_rmse_until_cross_acc_1_000_top_27_with_filter      ,knapsack_rmse_until_cross_acc_0_960_top_27_no_filter        ,knapsack_rmse_until_cross_acc_0_970_top_27_no_filter        ,knapsack_rmse_until_cross_acc_0_980_top_27_no_filter        ,knapsack_rmse_until_cross_acc_0_990_top_27_no_filter        ,knapsack_rmse_until_cross_acc_0_965_top_27_no_filter        ,knapsack_rmse_until_cross_acc_0_975_top_27_no_filter        ,knapsack_rmse_until_cross_acc_0_985_top_27_no_filter        ,knapsack_rmse_until_cross_acc_0_995_top_27_no_filter        ,knapsack_rmse_until_cross_acc_1_000_top_27_no_filter        ,knapsack_cross_acc_until_cross_acc_0_960_top_27_with_filter ,knapsack_cross_acc_until_cross_acc_0_970_top_27_with_filter ,knapsack_cross_acc_until_cross_acc_0_980_top_27_with_filter ,knapsack_cross_acc_until_cross_acc_0_990_top_27_with_filter ,knapsack_cross_acc_until_cross_acc_0_965_top_27_with_filter ,knapsack_cross_acc_until_cross_acc_0_975_top_27_with_filter ,knapsack_cross_acc_until_cross_acc_0_985_top_27_with_filter ,knapsack_cross_acc_until_cross_acc_0_995_top_27_with_filter ,knapsack_cross_acc_until_cross_acc_1_000_top_27_with_filter ,knapsack_cross_acc_until_cross_acc_0_960_top_27_no_filter   ,knapsack_cross_acc_until_cross_acc_0_970_top_27_no_filter   ,knapsack_cross_acc_until_cross_acc_0_980_top_27_no_filter   ,knapsack_cross_acc_until_cross_acc_0_990_top_27_no_filter   ,knapsack_cross_acc_until_cross_acc_0_965_top_27_no_filter   ,knapsack_cross_acc_until_cross_acc_0_975_top_27_no_filter   ,knapsack_cross_acc_until_cross_acc_0_985_top_27_no_filter   ,knapsack_cross_acc_until_cross_acc_0_995_top_27_no_filter   ,knapsack_cross_acc_until_cross_acc_1_000_top_27_no_filter   ], "pareto")

        #self.create_solver_plot(delta_0_4, ['choose_instances_random_cross_acc', 'choose_instances_random_true_acc_v2'], "avg plot true acc", '0_CaDiCaL_DVDL_V1')
        #self.create_average_plot(delta_0_4, ['choose_instances_random_cross_acc'], "avg plot cross acc")

        #self.create_average_plot(min_rmse_dont_break, ['determine_timeouts_stability'], "stability")
        #self.create_average_plot(min_rmse_dont_break, ['determine_timeouts_true_acc'], "min diff determine timeout true acc")
        #self.create_average_plot(min_rmse_dont_break, ['determine_timeouts_cross_acc'], "min diff determine timeout cross acc")


        #self.create_average_plot(knapsack_dont_break, ['determine_timeouts_true_acc'], "knap determine timeout true acc")
        #self.create_average_plot(knapsack_dont_break, ['determine_timeouts_cross_acc'], "knap determine timeout cross acc")
        
        #self.create_average_plot(min_rmse_dont_break, ['determine_timeouts_diff'], "knap determine timeout diff")
        #self.create_average_plot(min_rmse_dont_break, ['determine_timeouts_stability'], "min rmse determine timeout stability")
        #self.create_average_plot(min_cross_acc_dont_break, ['determine_timeouts_cross_acc_stability'], "cross acc determine timeout stability")
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

        #plt.plot(al_low_delta_rt, al_high_delta_acc, marker='o', linestyle='None', label='active learning')
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
        # old label: label=f"{label} (RMSE={rmse:.2f})"
        plt.plot(prediction, actual, marker='.', linestyle='None', label=label, color=color)

        # Plot regression line
        sorted_indices = np.argsort(prediction)
        plt.plot(np.array(prediction)[sorted_indices], predicted_line[sorted_indices], linestyle='-', color=color)

        return model.coef_[0], model.intercept_, rmse

    def visualize_predictions(self, df_rated: pd.DataFrame, df_runtimes):

        df_rated_copy = df_rated.copy()

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
        m5000, c5000, rmse5000 = self.plot_regression(actual, actual, r'$\mathcal{B}_{1}=(I, \tau_{5000})$', colors[0])
        m4000, c4000, rmse4000 = self.plot_regression(prediction_4000, actual, r'$\mathcal{B}_{2}=(I, \tau_{4000})$', colors[1])
        m3000, c3000, rmse3000 = self.plot_regression(prediction_3000, actual, r'$\mathcal{B}_{3}=(I, \tau_{3000})$', colors[2])
        m2000, c2000, rmse2000 = self.plot_regression(prediction_2000, actual, r'$\mathcal{B}_{4}=(I, \tau_{2000})$', colors[3])
        #m1000, c1000, rmse1000 = self.plot_regression(prediction_1000, actual, r'$\mathcal{B}=(I, \tau_{1000})$', colors[4])
        self.visualize_predictions_exclude_instances(df_rated_copy, df_runtimes)
        plt.legend()
        plt.xlabel(r'$s_{\mathcal{B}}(a)$', fontsize=16)
        plt.ylabel(r'$s_{\mathcal{B}_\mathrm{total}}(a)$', fontsize=16)
        plt.title(r'functional relationship between $s_{\mathcal{B}_\mathrm{total}}(a)$ and $s_\mathcal{B}(a)$')
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        # Show or save
        plt.show()

    def visualize_predictions_exclude_instances(self, df_rated: pd.DataFrame, df_runtimes: pd.DataFrame):

        par_2_scores_series = df_rated.mean(axis=0)

        print(par_2_scores_series)

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
        #df_rated_4000[df_rated_4000 > 1000] = 2000
        par_2_scores_4000 = df_rated_4000.mean(axis=0)

        mask_4000_th = instance_mean_rt > 4000
        df_rated_4000_th = df_runtimes.copy()
        df_rated_4000_th[mask_4000_th] = np.nan
        df_rated_4000_th[df_rated_4000_th > 1000] = 2000
        par_2_scores_4000_th = df_rated_4000_th.mean(axis=0)

        actual = []
        prediction_500 = []
        prediction_1000 = []
        prediction_3000 = []
        prediction_2000 = []
        prediction_4000 = []
        prediction_4000_th = []
        for solver in par_2_scores_series.index:
            actual.append(par_2_scores_series[solver])
            prediction_500.append(par_2_scores_500[solver])
            prediction_1000.append(par_2_scores_1000[solver])
            prediction_3000.append(par_2_scores_3000[solver])
            prediction_2000.append(par_2_scores_2000[solver])
            prediction_4000.append(par_2_scores_4000[solver])
            prediction_4000_th.append(par_2_scores_4000_th[solver])

        #plt.figure(figsize=(10, 6))
        #plt.plot(actual, actual, marker=".", linestyle='None', label="timeout=5000")
        colors = ['black', 'magenta', 'brown', 'olive', 'pink', 'purple', 'gray']
        #m100, c100, rmse100 = self.plot_regression(prediction_500, actual, "mean_time <= 500", colors[5])
        #m1000, c1000, rmse1000 = self.plot_regression(prediction_1000, actual, "mean_time <= 1000", colors[4])        
        m3000, c3000, rmse3000 = self.plot_regression(prediction_2000, actual, r'$\mathcal{B}_{5}=(I_{2000}, \tau_{5000})$', colors[2])
        #m2000, c2000, rmse2000 = self.plot_regression(prediction_3000, actual, "mean_time <= 3000", colors[3])
        m4000, c4000, rmse4000 = self.plot_regression(prediction_4000, actual, r'$\mathcal{B}_{6}=(I_{4000}, \tau_{5000})$', colors[1])
        m4000_th, c4000_th, rmse4000_th = self.plot_regression(prediction_4000_th, actual, r'$\mathcal{B}_{7}=(I_{4000}, \tau_{1000})$', colors[0])
        #m5000, c5000, rmse5000 = self.plot_regression(actual, actual, "mean_time <= inf", colors[0])

        #plt.legend()
        #plt.xlabel("mean_time limited")
        #plt.ylabel("mean_time unlimited")
        #plt.title("Comparision of reduced instance pool par-2-scores")
        #plt.grid(True, linestyle="--", alpha=0.5)
        #plt.tight_layout()
        # Show or save
        #plt.show()