#!/bin/bash
#SBATCH -A gaa_serv
#SBATCH -p gaa
#SBATCH -n 16
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --job-name=swiss_roll
#SBATCH --output=swiss_roll.out

# conda activate dm_nys

printf "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\nEXPERIMENT: SWISS ROLL\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
python -m experiments.swiss_roll.experiment -c experiments/swiss_roll/config.yml
python -m experiments.swiss_roll.plot_results -c experiments/swiss_roll/config.yml