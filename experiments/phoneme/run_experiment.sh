#!/bin/bash
#SBATCH -A gaa_serv
#SBATCH -p gaa
#SBATCH -n 16
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --job-name=phoneme
#SBATCH --output=phoneme.out

# conda activate dm_nys

printf "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\nEXPERIMENT: PHONEME\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
python -m experiments.phoneme.experiment -c experiments/phoneme/config.yml
python -m experiments.phoneme.plot_results -c experiments/phoneme/config.yml