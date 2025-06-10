#!/bin/bash
#SBATCH -J FCN_test       # job name, optional
#SBATCH -N 1          # number of computing node
#SBATCH --gres=gpu:1  # number of gpus allocated on each node
#SBATCH -w node04  # apply for node04
#SBATCH -o ../logs/FCN_test_ori.%j.out

b=64
lr=0.001
widths=(3)
depths=(3)

# alpha_train_list=(1.3)
# sigma_train_list=(0.3)
# alpha_test_list=(0 2 1.9 1.5 1.3 1 0.9 'multiple' 'mixture')
# sigma_test_list=(0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5)

# # Iterate through sigma_train for each alpha_train.
# for width in "${widths[@]}"; do
#   for depth in "${depths[@]}"; do
#     for alpha_train in "${alpha_train_list[@]}"; do
#       for sigma_train in "${sigma_train_list[@]}"; do
#         for alpha_test in "${alpha_test_list[@]}"; do
#           for sigma_test in "${sigma_test_list[@]}"; do
#             params="--b=$b --width=$width --depth=$depth --lr=$lr --alpha_train=$alpha_train --alpha_test=$alpha_test --sigma_train=$sigma_train --sigma_test=$sigma_test"
#             echo "Script started: $params"
#             python -u FCN_test.py $params 2>&1
#             echo "Script completed: $params"
#             if [ $alpha_test -eq 0 ]; then
#               break
#             fi
#           done
#         done
#         if [ $alpha_train -eq 0 ]; then
#           break
#         fi
#       done
#     done
#   done
# done

# Only run for the optimal pairs of alpha_train and sigma_train.
alpha_train_list=(0 2 1.9 1.5 1.3 1 0.9 'multiple' 'mixture')
sigma_train_list=(0.1 0.35 0.35 0.2 0.15 0.1 0.1 0.25 0.25)
alpha_test_list=(0 2 1.9 1.5 1.3 1 0.9 'multiple' 'mixture')
sigma_test_list=(0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5)
for width in "${widths[@]}"; do
  for depth in "${depths[@]}"; do
    for ((i=0; i<${#alpha_train_list[@]}; i++)) do
      alpha_train=${alpha_train_list[i]}
      sigma_train=${sigma_train_list[i]}
      for alpha_test in "${alpha_test_list[@]}"; do
        for sigma_test in "${sigma_test_list[@]}"; do
          params="--b=$b --width=$width --depth=$depth --lr=$lr --alpha_train=$alpha_train --alpha_test=$alpha_test --sigma_train=$sigma_train --sigma_test=$sigma_test"
          echo "Script started: $params"
          python -u FCN_test_ori.py $params 2>&1
          echo "Script completed: $params"
          if [ $alpha_test -eq 0 ]; then
            break
          fi
        done
      done
    done
  done
done