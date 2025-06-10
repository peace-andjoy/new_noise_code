#!/bin/bash
#SBATCH -J libras       # job name, optional
#SBATCH -N 1          # number of computing node
#SBATCH --gres=gpu:1  # number of gpus allocated on each node
#SBATCH -w node05  # apply for node04
#SBATCH -o ../logs/libras.%j.out

# 定义要运行的脚本参数
model='lstm_self'
b=32
num=5
optimizer='nadam'
lr=0.001
num_unitss=(128)
num_layerss=(3)
nb_iterations=1500
alpha_list=(0 2 1.9 1.5 1.3 1 0.9 'multiple' 'mixture')
sigma_list=(0.05 0.4 0.4 0.5 0.35 0.2 0.2 0.3 0.05)

# # Iterate through sigma_train for each alpha_train.
# for num_units in "${num_unitss[@]}"; do
#   for num_layers in "${num_layerss[@]}"; do
#     for alpha in "${alpha_list[@]}"; do
#       for sigma in "${sigma_list[@]}"; do
#         params="--dataset=libras --model=$model --b=$b --num=$num --optimizer=$optimizer --lr=$lr --num_units=$num_units --num_layers=$num_layers --alpha=$alpha --sigma=$sigma --nb_iterations=$nb_iterations"
#         echo "Script started: $params"
#         python -u ts_train.py $params 2>&1
#         echo "Script completed: $params"
#         if [ $alpha -eq 0 ]; then
#           break
#         fi
#       done
#     done
#   done
# done

# Only run for the optimal pairs of alpha_train and sigma_train.
for num_units in "${num_unitss[@]}"; do
  for num_layers in "${num_layerss[@]}"; do
    for ((i=0; i<${#alpha_list[@]}; i++)) do
      alpha=${alpha_list[i]}
      sigma=${sigma_list[i]}
      params="--dataset=libras --model=$model --b=$b --num=$num --optimizer=$optimizer --lr=$lr --num_units=$num_units --num_layers=$num_layers --alpha=$alpha --sigma=$sigma --nb_iterations=$nb_iterations"
      echo "Script started: $params"
      python -u ts_train.py $params 2>&1
      echo "Script completed: $params"
    done
  done
done

