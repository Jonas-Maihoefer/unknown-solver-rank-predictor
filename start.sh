# Set the number of threads for parallel math libraries
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16

# Activate virtual environment and run your application
source venv/bin/activate && sh run.sh