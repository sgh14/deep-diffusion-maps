#!/bin/bash
#SBATCH -A gaa_serv
#SBATCH -p gaa
#SBATCH -n 16
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --job-name=mnist
#SBATCH --output=mnist.out

# conda activate dm_nys

printf "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\nEXPERIMENT: MNIST\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
python -m experiments.mnist.hyperparameters_search -c experiments/mnist/config.yml
python -m experiments.mnist.experiment -c experiments/mnist/config.yml
python -m experiments.mnist.plot_results -c experiments/mnist/config.yml