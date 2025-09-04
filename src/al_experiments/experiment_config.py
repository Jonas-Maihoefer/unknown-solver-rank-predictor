class ExperimentConfig:

    def __init__(
            self,
            filter_unsolvable,
            determine_thresholds,
            select_idx,
            scoring_fn,
            thresh_breaking_conditions,
            temperatures,
            rt_weights,
            instance_selections,
            individual_solver_plots: bool
    ):
        self.filter_unsolvable = filter_unsolvable
        self.determine_thresholds = determine_thresholds
        self.select_idx = select_idx
        self.scoring_fn = scoring_fn
        self.thresh_breaking_conditions = thresh_breaking_conditions
        self.temperatures = temperatures
        self.rt_weights = rt_weights
        self.instance_selections = instance_selections
        self.individual_solver_plots = individual_solver_plots
