#!/bin/bash
#SBATCH -J libras       # job name, optional
#SBATCH -N 1          # number of computing node
#SBATCH --gres=gpu:1  # number of gpus allocated on each node
#SBATCH -w node05  # apply for node04
#SBATCH -o ../logs/libras_train_test.%j.out

model='lstm_self'
b=32
num=5
optimizer='nadam'
lr=0.001
sigma=0.15
num_unitss=(128)
num_layerss=(1 2 3)
alpha_list=(0 2 1.9 1.5 1.3 1 0.9 0.5 'multiple' 'mixture')
sigma_list=(0.01 0.2 0.25 0.2 0.09 0.09 0.07 0.02 0.15 0.03)

for num_units in "${num_unitss[@]}"; do
  for num_layers in "${num_layerss[@]}"; do
    for ((i=0; i<${#alpha_list[@]}; i++)) do
      alpha=${alpha_list[i]}
      sigma=${sigma_list[i]}
      params="--dataset=libras --model=$model --b=$b --num=$num --optimizer=$optimizer --lr=$lr --num_units=$num_units --num_layers=$num_layers --alpha=$alpha --sigma=$sigma"
      echo "Script started: $params"
      python -u ts_train.py $params 2>&1
      python -u ts_test.py $params 2>&1
      echo "Script completed: $params"
  done
done

model='lstm_self'
b=32
num=5
optimizer='nadam'
lr=0.001
sigma=0.15
num_unitss=(256 512)
num_layerss=(1)
alpha_list=(0 2 1.9 1.5 1.3 1 0.9 0.5 'multiple' 'mixture')
sigma_list=(0.01 0.2 0.25 0.2 0.09 0.09 0.07 0.02 0.15 0.03)

for num_units in "${num_unitss[@]}"; do
  for num_layers in "${num_layerss[@]}"; do
    for ((i=0; i<${#alpha_list[@]}; i++)) do
      alpha=${alpha_list[i]}
      sigma=${sigma_list[i]}
      params="--dataset=libras --model=$model --b=$b --num=$num --optimizer=$optimizer --lr=$lr --num_units=$num_units --num_layers=$num_layers --alpha=$alpha --sigma=$sigma"
      echo "Script started: $params"
      python -u ts_train.py $params 2>&1
      python -u ts_test.py $params 2>&1
      echo "Script completed: $params"
  done
done