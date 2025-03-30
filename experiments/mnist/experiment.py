import numpy as np
import os
import h5py
import argparse
import yaml
from tensorflow.keras.optimizers import Adam

from diffusionloss import DiffusionLoss
from diffusionmaps import DiffusionMaps
from experiments.aux_functions import get_sigma, log_likelihood_curve
from experiments.mnist.load_data import get_datasets
from experiments.metrics import distances_errors
from experiments.models import build_conv_encoder

import tensorflow as tf
print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))

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
# Copy configuration file
os.system(f'cp {args.config} {output_dir}')

# Get the data
(X_a, y_a), (X_b, y_b) = get_datasets(
    npoints=config['data']['npoints'],
    split=config['data']['split'],
    seed=config['data']['seed'],
    noise=config['data']['noise']
)
img_shape = X_a.shape[1:]
X_a = np.expand_dims(X_a, axis=-1)
X_b = np.expand_dims(X_b, axis=-1)
X = np.vstack([X_a, X_b])

# Compute sigma and initialize Diffusion Maps
sigma = get_sigma(X_a, q)
DM = DiffusionMaps(sigma=sigma, n_components=n_components, steps=steps, alpha=alpha)

# Approach 1: Original Diffusion Maps
X_red_1 = DM.fit_transform(X)
X_a_red_1 = X_red_1[:len(X_a)]
X_b_red_1 = X_red_1[len(X_a):]

# Approach 2: Diffusion Maps with Diffusion Loss
encoder = build_conv_encoder(
    input_shape=X_a.shape[1:],
    filters=config['encoder']['filters'],
    n_components=n_components,
    zero_padding=(2, 2),
    dropout=config['encoder']['dropout'],
    use_bn=config['encoder']['batch_normalization']
)
encoder.summary()
loss = DiffusionLoss(X_a, sigma=sigma, steps=steps, alpha=alpha)
optimizer = Adam(learning_rate=config['encoder']['learning_rate'])
encoder.compile(loss=loss, optimizer=optimizer)

indices = np.arange(X_a.shape[0])
hist_enc = encoder.fit(
    x=X_a,
    y=indices,
    epochs=config['encoder']['epochs'],
    validation_split=config['encoder']['validation_split'],
    shuffle=True,
    batch_size=config['encoder']['batch_size'],
    verbose=2
)
X_a_red_2 = encoder(X_a)
X_b_red_2 = encoder(X_b)

encoder.save(os.path.join(output_dir, 'encoder.keras'))
with h5py.File(os.path.join(output_dir, 'hist_enc.h5'), 'w') as file:
    for key, value in hist_enc.history.items():
        file.create_dataset(key, data=value)

results_2_a= distances_errors(X_a_red_1, X_a_red_2.numpy())
results_2_b = distances_errors(X_b_red_1, X_b_red_2.numpy())

# Approach 3: Nyström Extension
X_a_red_3 = DM.fit_transform(X_a)
eigenvalues = DM.lambdas[1:]**steps
log_likelihood = log_likelihood_curve(eigenvalues)
X_b_red_3 = DM.transform(X_b)
results_3_a = distances_errors(X_a_red_1, X_a_red_3)
results_3_b = distances_errors(X_b_red_1, X_b_red_3)

# Save results
X_a = X_a.reshape(-1, *img_shape)
X_b = X_b.reshape(-1, *img_shape)
with h5py.File(os.path.join(output_dir, 'results.h5'), "w") as file:
    # Save hyperparameters without compression (small scalars)
    file.create_group("hyperparameters")
    file['hyperparameters'].create_dataset("n_components", data=n_components)
    file['hyperparameters'].create_dataset("q", data=q)
    file['hyperparameters'].create_dataset("alpha", data=alpha)
    file['hyperparameters'].create_dataset("steps", data=steps)
    file['hyperparameters'].create_dataset("sigma", data=sigma)

    # Organize all data in a nested dictionary
    data = {
        'original': {
            'train': {'X': X_a, 'y': y_a},
            'test': {'X': X_b, 'y': y_b}
        },
        'diffusion_maps': {
            'train': {'X_red': X_a_red_1},
            'test': {'X_red': X_b_red_1}
        },
        'deep_diffusion_maps': {
            'train': {'X_red': X_a_red_2, **results_2_a},
            'test': {'X_red': X_b_red_2, **results_2_b}
        },
        'nystrom': {
            'train': {
                'X_red': X_a_red_3,
                'eigenvalues': eigenvalues,
                'log_likelihood': log_likelihood,
                **results_3_a
            },
            'test': {'X_red': X_b_red_3, **results_3_b}
        }
    }

    # Create the groups and datasets with compression for large arrays
    for group_name, group_dict in data.items():
        file.create_group(group_name)
        for subset_name, subset_dict in group_dict.items():
            file[group_name].create_group(subset_name)
            for key, value in subset_dict.items():
                if hasattr(data, 'shape') and len(data) > 1:
                    file[group_name][subset_name].create_dataset(key, data=value, compression='gzip')
                else:
                    file[group_name][subset_name].create_dataset(key, data=value)