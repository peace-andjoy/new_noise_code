#!/bin/bash
#SBATCH -J SimpleTest13       # job name, optional
#SBATCH -N 1          # number of computing node
#SBATCH --gres=gpu:1  # number of gpus allocated on each node
#SBATCH -w node04  # apply for node04
#SBATCH -o SimpleTest13.out

python SimpleTest.py