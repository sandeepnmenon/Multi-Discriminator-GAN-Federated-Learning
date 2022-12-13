#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name=idls_project

module purge

module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
conda activate idls
cd /scratch/snm6477/Multi-Discriminator-GAN-Federated-Learning/

./run_mnist_exp.sh
# python problem_4_env2.py

