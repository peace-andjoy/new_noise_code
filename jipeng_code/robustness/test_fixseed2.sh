#!/bin/bash
#SBATCH -J 2test_sigma      # job name, optional
#SBATCH -N 1          # number of computing node
#SBATCH -c 1          # number of cpus, for multi-thread programs
#SBATCH --gres=gpu:1  # number of gpus allocated on each node
#SBATCH -w node05  # apply for node05
#SBATCH -o 2test_sigma.out

# 定义要运行的脚本参数
params=(
    "--num_filters=16 --num_res=6 --sigma=0.05"
)

# 循环运行脚本
for param in "${params[@]}"; do
    python -u test_fixseed2.py $param 2>&1
    echo "Script completed: $param"
done
