#!/bin/bash
#SBATCH -J libras_test       # job name, optional
#SBATCH -N 1          # number of computing node
#SBATCH --gres=gpu:1  # number of gpus allocated on each node
#SBATCH -w node04  # apply for node04
#SBATCH -o ../logs/libras_test.%j.out


# # Iterate through sigma_train for each alpha_train.
# b=32
# lr=0.001
# num_unitss=(128)
# num_layerss=(1)
# nb_iterations=1500
# alpha_train_list=(2 1.9 1.5)
# sigma_train_list=(0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5)
# alpha_test_list=(0)
# sigma_test_list=(0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5)
# sigma_test_list=(0.05)
# for num_units in "${num_unitss[@]}"; do
#   for num_layers in "${num_layerss[@]}"; do
#     for alpha_train in "${alpha_train_list[@]}"; do
#       for sigma_train in "${sigma_train_list[@]}"; do
#         for alpha_test in "${alpha_test_list[@]}"; do
#           for sigma_test in "${sigma_test_list[@]}"; do
#             params="--b=$b --alpha_train=$alpha_train --sigma_train=$sigma_train --alpha_test=$alpha_test --sigma_test=$sigma_test --lr=$lr --nb_iterations=$nb_iterations --num_units=$num_units --num_layers=$num_layers"
#             echo "Script started: $params"
#             python -u ts_test.py $params 2>&1
#             echo "Script completed: $params"
#             if [ $alpha_test -eq 0 ]; then
#               break
#             fi
#           done
#         done
#         if [ $alpha_train -eq 0 ]; then
#             break
#         fi
#       done
#     done
#   done
# done

# Only run for the optimal pairs of alpha_train and sigma_train.
b=32
lr=0.001
num_unitss=(128)
num_layerss=(3)
nb_iterations=1500
alpha_train_list=(0.0 2.0 1.9 1.5 1.3 1.0 0.9 'multiple' 'mixture')
sigma_train_list=(0.05 0.4 0.4 0.5 0.35 0.2 0.2 0.3 0.05)
alpha_test_list=(0 2 1.9 1.5 1.3 1 0.9 0.7 'multiple' 'mixture')
sigma_test_list=(0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5)
for num_units in "${num_unitss[@]}"; do
  for num_layers in "${num_layerss[@]}"; do
    for ((i=0; i<${#alpha_train_list[@]}; i++)) do
      alpha_train=${alpha_train_list[i]}
      sigma_train=${sigma_train_list[i]}
      for alpha_test in "${alpha_test_list[@]}"; do
        for sigma_test in "${sigma_test_list[@]}"; do
          params="--b=$b --alpha_train=$alpha_train --sigma_train=$sigma_train --alpha_test=$alpha_test --sigma_test=$sigma_test --lr=$lr --nb_iterations=$nb_iterations --num_units=$num_units --num_layers=$num_layers"
          echo "Script started: $params"
          python -u ts_test.py $params 2>&1
          echo "Script completed: $params"
          if [ $alpha_test -eq 0 ]; then
            break
          fi
        done
      done
    done
  done
done