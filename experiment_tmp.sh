#!/bin/bash
#SBATCH -J exp       # job name, optional
#SBATCH -N 1          # number of computing node
#SBATCH --gres=gpu:1  # number of gpus allocated on each node
#SBATCH -w node05  # apply for node04
#SBATCH -o ../logs/exp.%j.out

# 定义要运行的脚本参数
widths=(10)
depths=(3)
num=5
# alpha_list=(0 2 1.9 1.5 1.3 1 0.9 0.5 'multiple' 'mixture')
alpha_list=(2)
# sigma_list=(0.1 0.25 0.35 0.2 0.15 0.1 0.1 0.05 0.25 0.25)
sigma_list=(0.25)
lr=0.001
iter=10000

# 循环运行脚本
for width in "${widths[@]}"; do
  for depth in "${depths[@]}"; do
    for ((i=0; i<${#alpha_list[@]}; i++)) do
      alpha=${alpha_list[i]}
      sigma=${sigma_list[i]}
      params="--b=64 --width=$width --depth=$depth --num=$num --lr=$lr --iter=$iter --alpha=$alpha --sigma=$sigma"
      echo "Script started: $params"
      python -u FCN_train.py $params 2>&1
      echo "Script completed: $params"
    done
  done
done