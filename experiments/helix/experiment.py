import time
import numpy as np
import os
import h5py
import argparse
import yaml
from tensorflow.keras.optimizers import Adam

from DiffusionLoss import DiffusionLoss
from DiffusionMaps import DiffusionMaps
from experiments.aux_functions import get_sigma, plot_loglikelihood, plot_eigenvalues
from experiments.helix.load_data import get_datasets, my_colormap1D, normalize
from experiments.metrics import distances_mae_and_mrae
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
    npoints=config['data']['npoints'],
    split=config['data']['split'],
    seed=config['data']['seed'],
    noise=config['data']['noise']
)
X = np.vstack([X_a, X_b])

# Plot eigenvalues and log-likelihood
plot_eigenvalues(X_a, [q], [alpha], [steps], output_dir, max_components=25)
plot_loglikelihood(X_a, [q], [alpha], [steps], output_dir, max_components=25)

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
tic = time.perf_counter()
loss = DiffusionLoss(X_a, sigma=sigma, steps=steps, alpha=alpha)
optimizer = Adam(learning_rate=config['encoder']['learning_rate'])
encoder.compile(loss=loss, optimizer=optimizer)

indices = np.arange(X_a.shape[0])
hist_enc = encoder.fit(
    x=X_a,
    y=indices,
    epochs=config['encoder']['epochs'],
    validation_split=config['encoder']['validation_split'],
    shuffle=False,
    batch_size=config['encoder']['batch_size'],
    verbose=2
)
X_a_red_2 = encoder(X_a)
tac = time.perf_counter()
X_b_red_2 = encoder(X_b)
toc = time.perf_counter()

encoder.save(os.path.join(output_dir, 'encoder.keras'))
with h5py.File(os.path.join(output_dir, 'hist_enc.h5'), 'w') as file:
    for key, value in hist_enc.history.items():
        file.create_dataset(key, data=value)

(mae_2, mae_2_ci), (mrae_2, mrae_2_ci) = distances_mae_and_mrae(X_b_red_1, X_b_red_2.numpy())

# Approach 3: Nyström Extension
X_a_red_3 = DM.fit_transform(X_a)
X_b_red_3 = DM.transform(X_b)
(mae_3, mae_3_ci), (mrae_3, mrae_3_ci) = distances_mae_and_mrae(X_b_red_1, X_b_red_3)

P_a = DM.transition_probabilities(DM.W)
if steps > 1:
    P_a = np.linalg.matrix_power(P_a, steps)

D_a = DM.diffusion_distances(P_a, DM.pi)
point = np.argmin(np.linalg.norm(X_a - np.array([0, -1, -15]), axis=1))
P_color_a = np.array([my_colormap1D(x, c1=(0, 0, 0), c2=(0, 0.75, 0.75)) for x in normalize(P_a[:, point])])
D_color_a = np.array([my_colormap1D(x, c1=(0, 0, 0), c2=(0.75, 0, 0.75)) for x in normalize(D_a[:, point])])
P_color_a[point] = (0.72, 0.53, 0.05) # (0.75, 0, 0.75) 
D_color_a[point] = (0.72, 0.53, 0.05) # (0, 0.75, 0.75)

# Save results
with h5py.File(os.path.join(output_dir, 'results.h5'), "w") as file:
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
    group_2.create_dataset("mae_conf_int", data=mae_2_ci)
    group_2.create_dataset("mrae", data=mrae_2)
    group_2.create_dataset("mrae_conf_int", data=mrae_2_ci)

    group_3 = file.create_group("nystrom")
    group_3.create_dataset("X_a_red", data=X_a_red_3, compression='gzip')
    group_3.create_dataset("X_b_red", data=X_b_red_3, compression='gzip')
    group_3.create_dataset("mae", data=mae_3)
    group_3.create_dataset("mae_conf_int", data=mae_3_ci)
    group_3.create_dataset("mrae", data=mrae_3)
    group_3.create_dataset("mrae_conf_int", data=mrae_3_ci)
    group_3.create_dataset("P_color_a", data=P_color_a, compression='gzip')
    group_3.create_dataset("D_color_a", data=D_color_a, compression='gzip')
