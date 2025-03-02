import numpy as np
import os
import argparse
import yaml

from experiments.aux_functions import plot_eigenvalues, plot_loglikelihood, plot_distance_percentiles, plot_intrinsic_dimension_estimation
from experiments.phoneme.load_data import get_datasets


# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, required=True, help="Path to the YAML configuration file.")
args = parser.parse_args()

# Load Configuration
with open(args.config, "r") as file:
    config = yaml.safe_load(file)

# Extract parameters from config file
root = config['output_dir']
max_components = config['hyperparameters_search']['max_components']
q_vals = config['hyperparameters_search']['quantile_values']
alpha_vals = config['hyperparameters_search']['alpha_values']
steps_vals = config['hyperparameters_search']['steps_values']

# Create directories
output_dir = root
os.makedirs(output_dir, exist_ok=True)

# Get the data
(X_a, y_a), (X_b, y_b) = get_datasets(
    split=config['data']['split'],
    seed=config['data']['seed'],
    noise=config['data']['noise']
)
X_a = np.reshape(X_a, (X_a.shape[0], -1))
X_b = np.reshape(X_b, (X_b.shape[0], -1))
X = np.vstack([X_a, X_b])

plot_eigenvalues(X_a, q_vals, alpha_vals, steps_vals, output_dir, max_components)
plot_loglikelihood(X_a, q_vals, alpha_vals, steps_vals, output_dir, max_components)
# plot_distance_percentiles(X_a, q_vals, alpha_vals, steps_vals, output_dir)
plot_intrinsic_dimension_estimation(X_a, k_max=X_a.shape[0]//2, output_dir=output_dir)
