#!/bin/bash

# ==============================================================================
# 🚀 Accelerating Decision Diffuser - Ultimate Baseline Launcher
# ==============================================================================

# 1. Hardware & Simulation Environment Setup
export MUJOCO_GL="osmesa"
export PYOPENGL_PLATFORM="osmesa"

# Ensure results directory exists
mkdir -p results

echo "========================================================"
echo "🚀 [1/2] 开始训练：Hopper-v2 (Medium-Expert)"
echo "========================================================"
# We remove specific overrides to use the optimized defaults in config.yaml
numactl --cpunodebind=0 --membind=0 python train.py env=hopper

echo "========================================================"
echo "🚀 [2/2] 开始训练：HalfCheetah-v2 (Medium-Expert)"
echo "========================================================"
numactl --cpunodebind=0 --membind=0 python train.py env=halfcheetah

echo "✅ 任务全部完成！结果已保存在 results/ 目录下。"
