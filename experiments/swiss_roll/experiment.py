import numpy as np
import os
import h5py
import argparse
import yaml
from tensorflow.keras.optimizers import Adam

from diffusionloss import DiffusionLoss
from diffusionmaps import DiffusionMaps
from experiments.aux_functions import get_sigma, log_likelihood_curve
from experiments.swiss_roll.load_data import get_datasets, my_colormap1D, normalize
from experiments.metrics import distances_errors
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
# Copy configuration file
os.system(f'cp {args.config} {output_dir}')

# Get the data
(X_a, y_a), (X_b, y_b) = get_datasets(
    npoints=config['data']['npoints'],
    split=config['data']['split'],
    seed=config['data']['seed'],
    noise=config['data']['noise']
)
X = np.vstack([X_a, X_b])

# Compute sigma and initialize Diffusion Maps
sigma = get_sigma(X_a, q)
DM = DiffusionMaps(sigma=sigma, n_components=n_components, steps=steps, alpha=alpha)

# Approach 1: Original Diffusion Maps
X_red_1 = DM.fit_transform(X)
X_a_red_1 = X_red_1[:len(X_a)]
X_b_red_1 = X_red_1[len(X_a):]

# Approach 2: Diffusion Maps with Diffusion Loss
encoder = build_encoder(
    input_shape=(X_a.shape[-1],),
    units=config['encoder']['units'],
    n_components=n_components,
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

P_a = DM.transition_probabilities(DM.W)
if steps > 1:
    P_a = np.linalg.matrix_power(P_a, steps)

D_a = DM.diffusion_distances(P_a, DM.pi)
point = np.argmin(np.linalg.norm(X_a - np.array([0, -1, -15]), axis=1))
P_color_a = np.array([my_colormap1D(x, c1=(0, 0, 0), c2=(0, 0.75, 0.75)) for x in normalize(P_a[:, point])])
D_color_a = np.array([my_colormap1D(x, c1=(0, 0, 0), c2=(0.75, 0, 0.75)) for x in normalize(D_a[:, point])])
P_color_a[point] = (0.72, 0.53, 0.05)
D_color_a[point] = (0.72, 0.53, 0.05)


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
                'P_color': P_color_a,
                'D_color': D_color_a,
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