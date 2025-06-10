#!/bin/bash
#SBATCH -J mnist_c       # job name, optional
#SBATCH -N 1          # number of computing node
#SBATCH --gres=gpu:1  # number of gpus allocated on each node
#SBATCH -w node05  # apply for node04
#SBATCH -o ../logs/mnist_c.%j.out


# 定义要运行的脚本参数
# alphas=(2 1.9 1.5 1.3 1 0.9 0.5 'mixture' 'multiple')
alphas=(1.5 1.3 1 0.9 0.5 'mixture' 'multiple')
num=5
# sigmas=(0.25 0.35 0.2 0.15 0.1 0.1 0.05 0.25 0.25)
sigmas=(0.2 0.15 0.1 0.1 0.05 0.25 0.25)

# 使用单个 for 循环遍历 alpha 和 sigma 数组，并获取对应的取值对
for ((i=0; i<${#alphas[@]}; i++))
do
  alpha=${alphas[i]}
  sigma=${sigmas[i]}
  # 在这里执行你想要的操作，例如调用其他命令或执行其他操作
  echo "alpha = $alpha, sigma = $sigma"
  params="--alpha=$alpha --sigma=$sigma"
  python -u mnist_c.py $params 2>&1
  # 在这里添加你想要执行的命令或操作
done