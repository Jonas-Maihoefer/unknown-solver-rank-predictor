import os


useCupy = os.getenv("USECUDA", "0") == "1"

if useCupy:
    import cupy as np
else:
    import numpy as np

number_of_solvers = 28
number_of_reduced_solvers = number_of_solvers-1
solver_fraction = 1/number_of_solvers
square_of_solvers = number_of_solvers * number_of_solvers
solver_pairs = number_of_solvers*(number_of_solvers-1)
reduced_solver_pairs = (number_of_reduced_solvers * (number_of_reduced_solvers - 1))
number_of_instances = 5355
instance_idx = np.arange(number_of_instances)

# indexes of sorted_runtimes tuple
idx = 0
rt = 1
