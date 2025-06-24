#!/bin/sh

# Set the number of threads for parallel math libraries
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16

# Use Cuda
export USECUDA=1

# load cuda package
spack load cuda@12.6.2

# 1) Ask Spack where CUDA@12.6.2 lives
CUDA_PREFIX=$(spack location -i cuda@12.6.2)

# 2) Check that this is the right directory
echo "CUDA prefix: $CUDA_PREFIX"
ls $CUDA_PREFIX/lib64/libnvrtc.so*

# 3) Export it into LD_LIBRARY_PATH (do this after you activate your venv)
export LD_LIBRARY_PATH=$CUDA_PREFIX/lib64:$LD_LIBRARY_PATH

# 4) Confirm
echo $LD_LIBRARY_PATH

# Activate virtual environment and run your application
source venv/bin/activate
sh run.sh