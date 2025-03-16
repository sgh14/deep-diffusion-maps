#!/bin/bash
#SBATCH -A gaa_serv
#SBATCH -p gaa
#SBATCH -n 16
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --job-name=s_curve
#SBATCH --output=s_curve.out

# Activate your conda environment
source ~/miniconda3/bin/activate
conda activate ddm

printf "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\nEXPERIMENT: S CURVE\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
python -m experiments.s_curve.experiment -c experiments/s_curve/config.yml
python -m experiments.s_curve.plot_results -c experiments/s_curve/config.yml