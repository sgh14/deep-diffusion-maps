import time
import numpy as np
import os
import h5py
import argparse
import yaml
from tensorflow.keras.optimizers import Adam

from DiffusionLoss import DiffusionLoss
from DiffusionMaps import DiffusionMaps
from experiments.aux_functions import get_sigma
from experiments.s_curve.load_data import get_datasets
from experiments.models import build_encoder


# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, required=True, help="Path to the YAML configuration file.")
args = parser.parse_args()

# Load Configuration
with open(args.config, "r") as file:
    config = yaml.safe_load(file)

# Extract parameters from config file
root = config['output_dir']
n_components = config['diffusion_maps']['n_components']
q = config['diffusion_maps']['quantile']
alpha = config['diffusion_maps']['alpha']
steps = config['diffusion_maps']['steps']

# Create directories
experiment = f'n_components_{n_components}_q_{q}_alpha_{alpha}_steps_{steps}'
output_dir = os.path.join(root, experiment)
os.makedirs(output_dir, exist_ok=True)

# Get the data
(X_a, y_a), (X_b, y_b) = get_datasets(
    npoints=100000,
    split=config['data']['split'],
    seed=config['data']['seed'],
    noise=config['data']['noise']
)
X = np.vstack([X_a, X_b])
npoints_vals = [100, 1000, 10000, 100000]
dm_times = []
ddm_train_times = []
ddm_test_times = []
nystrom_train_times = []
nystrom_test_times = []
for npoints in npoints_vals:
    X_sample = X[:npoints]
    # Compute sigma and initialize Diffusion Maps
    sigma = get_sigma(X_sample, q)
    DM = DiffusionMaps(sigma=sigma, n_components=n_components, steps=steps, alpha=alpha)
    # Approach 1: Original Diffusion Maps
    tic = time.perf_counter()
    _ = DM.fit_transform(X_sample)
    toc = time.perf_counter()
    dm_times.append(toc - tic)

    # Approach 2: Diffusion Maps with Diffusion Loss
    encoder = build_encoder(
        input_shape=(X_sample.shape[-1],),
        units=config['encoder']['units'],
        n_components=n_components,
        use_bn=config['encoder']['batch_normalization']
    )
    tic = time.perf_counter()
    loss = DiffusionLoss(X_sample, sigma=sigma, steps=steps, alpha=alpha)
    optimizer = Adam(learning_rate=config['encoder']['learning_rate'])
    encoder.compile(loss=loss, optimizer=optimizer)
    indices = np.arange(X_sample.shape[0])
    _ = encoder.fit(
        x=X_sample,
        y=indices,
        epochs=config['encoder']['epochs'],
        validation_split=config['encoder']['validation_split'],
        shuffle=False,
        batch_size=config['encoder']['batch_size'],
        verbose=2
    )
    _ = encoder(X_sample)
    tac = time.perf_counter()
    _ = encoder(X_sample)
    toc = time.perf_counter()
    ddm_train_times.append(tac - tic)
    ddm_test_times.append(toc - tac)

    # Approach 3: Nyström Extension
    tic = time.perf_counter()
    _ = DM.fit_transform(X_sample)
    tac = time.perf_counter()
    _ = DM.transform(X_sample)
    toc = time.perf_counter()
    nystrom_train_times.append(tac - tic)
    nystrom_test_times.append(toc - tac)

# Save results
with h5py.File(os.path.join(output_dir, 'results.h5'), "w") as file:
    file.create_dataset("tic", data=tic)
    group_hyperparameters = file.create_group("hyperparameters")
    group_hyperparameters.create_dataset("n_components", data=n_components)
    group_hyperparameters.create_dataset("q", data=q)
    group_hyperparameters.create_dataset("alpha", data=alpha)
    group_hyperparameters.create_dataset("steps", data=steps)
    group_hyperparameters.create_dataset("sigma", data=sigma)

    group_0 = file.create_group("original")
    group_0.create_dataset("X_a", data=X_a, compression='gzip')
    group_0.create_dataset("y_a", data=y_a, compression='gzip')
    group_0.create_dataset("X_b", data=X_b, compression='gzip')
    group_0.create_dataset("y_b", data=y_b, compression='gzip')

    group_1 = file.create_group("difussion_maps")
    group_1.create_dataset("X_a_red", data=X_a_red_1, compression='gzip')
    group_1.create_dataset("X_b_red", data=X_b_red_1, compression='gzip')

    group_2 = file.create_group("deep_diffusion_maps")
    group_2.create_dataset("X_a_red", data=X_a_red_2, compression='gzip')
    group_2.create_dataset("X_b_red", data=X_b_red_2, compression='gzip')
    group_2.create_dataset("mae", data=mae_2)

    group_3 = file.create_group("nystrom")
    group_3.create_dataset("X_a_red", data=X_a_red_3, compression='gzip')
    group_3.create_dataset("X_b_red", data=X_b_red_3, compression='gzip')
    group_3.create_dataset("mae", data=mae_3)
