#!/bin/bash

# 1. 强固物理仿真与渲染防线（脚本内多写一次，双重保险）
export LD_LIBRARY_PATH=/home/zihan_xu/.mujoco/mujoco210/bin:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export MUJOCO_GL="osmesa"
export PYOPENGL_PLATFORM="osmesa"

# 创建权重输出目录（防止代码里没创建导致保存报错）
mkdir -p outputs/weights

echo "========================================================"
echo "🚀 [1/2] 开始轰鸣：Hopper-v2 满血版训练正式点火！"
echo "========================================================"
numactl --cpunodebind=0 --membind=0 python train.py \
    env=hopper \
    epochs=200 \
    train_batch_size=32

echo "========================================================"
echo "🚀 [2/2] 阵地转移：HalfCheetah-v2 满血版训练无缝接力！"
echo "========================================================"
numactl --cpunodebind=0 --membind=0 python train.py \
    env=halfcheetah \
    epochs=200 \
    train_batch_size=32

echo "🎉 全线大获全胜！所有环境训练已全部收网！"
