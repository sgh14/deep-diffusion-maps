# Activate your conda environment
# source ~/miniconda3/bin/activate
# conda activate ddm

printf "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\nEXPERIMENT: HELIX\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
python -m experiments.helix.experiment -c experiments/helix/config.yml
python -m experiments.helix.plot_results -c experiments/helix/config.yml