#!/bin/bash

# Configuration
ENV_NAME="decision_diffuser"
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Activate environment
conda activate $ENV_NAME

# Performance environment variables for AMD EPYC
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export KMP_AFFINITY=granularity=fine,compact,1,0

# Execution with NUMA control
# Confining to socket 0 (first 256 threads) but further restricted to 32 logical cores by optimizer
echo "Starting Baseline Training on Hopper-v2..."
numactl --cpunodebind=0 --membind=0 python train.py env=hopper

echo "Starting Baseline Training on HalfCheetah-v2..."
numactl --cpunodebind=0 --membind=0 python train.py env=halfcheetah
