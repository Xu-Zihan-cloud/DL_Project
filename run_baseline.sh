#!/bin/bash

# ==============================================================================
# 🚀 Accelerating Decision Diffuser - Ultimate Baseline Launcher (Automated)
# ==============================================================================

# 1. Disable all interactive prompts (Crucial for unsupervised runs)
export WANDB_MODE=offline  # Lock to offline mode; prevents login/account selection prompts
export WANDB_SILENT=true   # Hide redundant wandb terminal messages

# 2. Hardware & Simulation Environment Setup
# Explicitly set MuJoCo paths for mujoco-py
export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export MUJOCO_GL="osmesa"
export PYOPENGL_PLATFORM="osmesa"

# Verify MuJoCo path exists to prevent silent failures
if [ ! -d "$HOME/.mujoco/mujoco210" ]; then
    echo "❌ Error: MuJoCo 2.1.0 not found at $HOME/.mujoco/mujoco210"
    exit 1
fi

# Ensure results directory exists
mkdir -p results

echo "========================================================"
echo "🚀 [1/2] 开始训练：Hopper-v2 (Medium-Expert)"
echo "========================================================"
# Use 'python -u' for unbuffered output, ensuring logs are real-time
numactl --cpunodebind=0 --membind=0 python -u train.py env=hopper

echo "========================================================"
echo "🚀 [2/2] 开始训练：HalfCheetah-v2 (Medium-Expert)"
echo "========================================================"
numactl --cpunodebind=0 --membind=0 python -u train.py env=halfcheetah

echo "✅ 任务全部完成！所有结果已安全保存在 results/ 目录下。"
