#!/bin/bash
#SBATCH -A gaa_serv
#SBATCH -p gaa
#SBATCH -n 16
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --job-name=helix
#SBATCH --output=helix.out

# Activate your conda environment
source ~/miniconda3/bin/activate
conda activate ddm

printf "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\nEXPERIMENT: HELIX\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
python -m experiments.helix.experiment -c experiments/helix/config.yml
python -m experiments.helix.plot_results -c experiments/helix/config.yml