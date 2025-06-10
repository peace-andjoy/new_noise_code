#!/bin/bash
#SBATCH -J single1      # job name, optional
#SBATCH -N 1          # number of computing node
#SBATCH -c 1          # number of cpus, for multi-thread programs
#SBATCH --gres=gpu:1  # number of gpus allocated on each node
#SBATCH -w node05  # apply for node05
#SBATCH -o single1.out

# 定义要运行的脚本参数
params=(
    # "--num_filters=8 --num_res=6"
    # "--num_filters=16 --num_res=5"
    "--num_filters=16 --num_res=6"
    # "--num_filters=16 --num_res=7"
    # "--num_filters=32 --num_res=6"
)

# 循环运行脚本
for param in "${params[@]}"; do
    python -u train_all1.py $param 2>&1
    echo "Script completed: $param"
done
