import os


useCupy = os.getenv("USECUDA", "0") == "1"

if useCupy:
    import cupy as np
else:
    import numpy as np

# indexes of sorted_runtimes tuple
idx = 0
rt = 1


class Constants:
    def __init__(self, df):
        self.number_of_solvers = df.shape[1]
        self.number_of_reduced_solvers = self.number_of_solvers-1
        self.solver_fraction = 1/self.number_of_solvers
        self.square_of_solvers = (
            self.number_of_solvers * self.number_of_solvers
        )
        self.solver_pairs = self.number_of_solvers*(self.number_of_solvers-1)
        self.reduced_solver_pairs = (
            self.number_of_reduced_solvers * (
                self.number_of_reduced_solvers - 1
            )
        )
        self.number_of_instances = df.shape[0]
        self.instance_idx = np.arange(self.number_of_instances)
