# Activate your conda environment
# source ~/miniconda3/bin/activate
# conda activate ddm

printf "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\nEXPERIMENT: MNIST\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
python -m experiments.mnist.experiment -c experiments/mnist/config.yml
python -m experiments.mnist.plot_results -c experiments/mnist/config.yml