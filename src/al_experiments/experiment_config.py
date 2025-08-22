class ExperimentConfig:

    def __init__(
            self,
            determine_thresholds,
            select_idx,
            scoring_fn,
            thresh_breaking_condition,
            temperatures,
            rt_weights,
            instance_selections,
            individual_solver_plots: bool
    ):
        self.determine_thresholds = determine_thresholds
        self.select_idx = select_idx
        self.scoring_fn = scoring_fn
        self.thresh_breaking_condition = thresh_breaking_condition
        self.temperatures = temperatures
        self.rt_weights = rt_weights
        self.instance_selections = instance_selections
        self.individual_solver_plots = individual_solver_plots
