#!/bin/sh
# Activate your conda environment
# source ~/miniconda3/bin/activate
# conda activate ddm

printf "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\nEXPERIMENT: S CURVE\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
python -m experiments.s_curve.experiment -c experiments/s_curve/config.yml
python -m experiments.s_curve.plot_results -c experiments/s_curve/config.yml