#!/bin/bash
#SBATCH -A gaa_serv
#SBATCH -p gaa
#SBATCH -n 16
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --job-name=mnist
#SBATCH --output=mnist.out

# Activate your conda environment
source ~/miniconda3/bin/activate
conda activate ddm

# module load cuda/12.0
# module load cudnn/8.0-cuda-11.0

printf "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\nEXPERIMENT: MNIST\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
python -m experiments.mnist.experiment -c experiments/mnist/config.yml
python -m experiments.mnist.plot_results -c experiments/mnist/config.yml