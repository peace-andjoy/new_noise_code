#!/bin/bash
#SBATCH -J exp       # job name, optional
#SBATCH -N 1          # number of computing node
#SBATCH --gres=gpu:1  # number of gpus allocated on each node
#SBATCH -w node05  # apply for node04
#SBATCH -o ../logs/FCN_train.%j.out


# # 定义要运行的脚本参数
# widths=(3)
# depths=(3)
# alphas=(0.7)
# num=5
# iter=10000
# sigmas=(0.05 0.1 0.15 0.2 0.25 0.3 0.35)

# # Iterate through sigma_train for each alpha_train.
# for width in "${widths[@]}"; do
#   for depth in "${depths[@]}"; do
#     for alpha in "${alphas[@]}"; do
#       for sigma in "${sigmas[@]}"; do
#         params="--b=64 --width=$width --depth=$depth --num=$num --lr=0.001 --iter=$iter --alpha=$alpha --sigma=$sigma"
#         echo "Script started: $params"
#         python -u FCN_train.py $params 2>&1
#         echo "Script completed: $params"
#         if [ $alpha -eq 0 ]; then
#           break
#         fi
#       done
#     done
#   done
# done

widths=(3)
depths=(3)
# alphas=(0 2 1.9 1.5 1.3 1 0.9 'multiple' 'mixture')
# sigmas=(0.1 0.35 0.35 0.3 0.3 0.2 0.15 0.25 0.25)
alphas=(1 'multiple')
sigmas=(0.2 0.25)
beta=0.25
num=1
iter=10000
lr=0.001
# Only run for the optimal pairs of alpha_train and sigma_train.
for width in "${widths[@]}"; do
  for depth in "${depths[@]}"; do
    for ((i=0; i<${#alphas[@]}; i++)) do
      alpha=${alphas[i]}
      sigma=${sigmas[i]}
      params="--b=64 --width=$width --depth=$depth --num=$num --lr=$lr --iter=$iter --alpha=$alpha --sigma=$sigma --beta=$beta"
      echo "Script started: $params"
      python -u FCN_train.py $params 2>&1
      echo "Script completed: $params"
    done
  done
done


