# Activate your conda environment
# source ~/miniconda3/bin/activate
# conda activate ddm

printf "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\nEXPERIMENT: PHONEME\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
python -m experiments.phoneme.experiment -c experiments/phoneme/config.yml
python -m experiments.phoneme.plot_results -c experiments/phoneme/config.yml