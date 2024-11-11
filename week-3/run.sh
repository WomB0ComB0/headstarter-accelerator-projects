#!/bin/bash

# Unset any CUDA related environment variables
unset CUDA_VISIBLE_DEVICES
unset LD_LIBRARY_PATH

# Set TensorFlow CPU configurations
export TF_CPP_MIN_LOG_LEVEL=2
export TF_ENABLE_ONEDNN_OPTS=0
export PROTOKERNEL_CPU_FEATURE_GUARD=0

# Run with minimal thread usage
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Run the application
python3 -W ignore main.py
