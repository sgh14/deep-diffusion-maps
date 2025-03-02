#!/bin/bash
#SBATCH -A gaa_serv
#SBATCH -p gaa
#SBATCH -n 16
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --job-name=helix
#SBATCH --output=helix.out

# conda activate dm_nys

printf "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\nEXPERIMENT: HELIX\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
python -m experiments.helix.hyperparameters_search -c experiments/helix/config.yml
python -m experiments.helix.experiment -c experiments/helix/config.yml
python -m experiments.helix.plot_results -c experiments/helix/config.yml