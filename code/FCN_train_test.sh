#!/bin/bash
#SBATCH -J FCN_train_test       # job name, optional
#SBATCH -N 1          # number of computing node
#SBATCH -w node04  # apply for node04
#SBATCH -o ../logs/FCN_test.%j.out

b=64
widths=(3)
depths=(3)
num=5
lr=0.001
iter=10000
# alpha_list=(0 2 1.9 1.5 1.3 1 0.9 0.5 'multiple' 'mixture')
# sigma_list=(0.1 0.25 0.35 0.2 0.15 0.1 0.1 0.05 0.25 0.25)
alpha_list=(1 'multiple')
sigma_list=(0.1 0.25)
beta=0.25

# 循环运行脚本
for width in "${widths[@]}"; do
  for depth in "${depths[@]}"; do
    for ((i=0; i<${#alpha_list[@]}; i++)) do
      alpha=${alpha_list[i]}
      sigma=${sigma_list[i]}
      params="--b=b --width=$width --depth=$depth --num=$num --lr=$lr --iter=$iter --alpha=$alpha --sigma=$sigma --beta=$beta"
      echo "Script started: $params"
      python -u FCN_train.py $params 2>&1
      python -u FCN_test.py $params 2>&1
      echo "Script completed: $params"
  done
done

# 定义要运行的脚本参数
b=64
widths=(3)
depths=(4 5)
num=5
sigma=0.2
lr=0.0005
iter=50000
alpha_list=(0 2 1.9 1.5 1.3 1 0.9 0.5 'multiple' 'mixture')
sigma_list=(0.1 0.25 0.35 0.2 0.15 0.1 0.1 0.05 0.25 0.25)

# 循环运行脚本
for width in "${widths[@]}"; do
  for depth in "${depths[@]}"; do
    for ((i=0; i<${#alpha_list[@]}; i++)) do
      alpha=${alpha_list[i]}
      sigma=${sigma_list[i]}
      params="--b=b --width=$width --depth=$depth --num=$num --lr=$lr --iter=$iter --alpha=$alpha --sigma=$sigma"
      echo "Script started: $params"
      python -u FCN_train.py $params 2>&1
      python -u FCN_test.py $params 2>&1
      echo "Script completed: $params"
  done
done