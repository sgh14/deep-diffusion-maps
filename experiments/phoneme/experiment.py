import time
import numpy as np
import os
import h5py
import tensorflow as tf

from DiffusionLoss import DiffusionLoss
from DiffusionMaps import DiffusionMaps
from experiments.aux_functions import get_sigma, find_optimal_hyperparameters, plot_eigenvalues
from experiments.phoneme.load_data import get_datasets
from experiments.metrics import mae
from experiments.models import build_seq_encoder

output_dir = '/scratch/sgarcia/ddm/experiments/phoneme/results'
os.makedirs(output_dir, exist_ok=True)

# Get the data
(X_a, y_a), (X_b, y_b) = get_datasets(split=0.222, seed=123, noise=0)
X = np.vstack([X_a, X_b])

# Find optimal values for n_components, q, steps and alpha
q_vals = np.array([0.01, 0.1, 0.2])
alpha_vals = np.array([0])
steps_vals = np.array([2**i for i in range(7)])
# plot_eigenvalues(X_a, q_vals, alpha_vals, steps_vals, output_dir, max_components=25)
# n_components, q, alpha, steps = find_optimal_hyperparameters(X_a, q_vals, alpha_vals, steps_vals, output_dir, max_components=25)
n_components, q, alpha, steps = 2, 1e-2, 0, 1
# values, counts = np.unique(y_a, return_counts=True)
sigma = get_sigma(X_a, q) # [y_a == values[np.argmin(counts)]]
DM = DiffusionMaps(sigma=sigma, n_components=n_components, steps=steps, alpha=alpha)

# Approach 1: original Diffusion Maps
X_red_1 = DM.fit_transform(X)
X_a_red_1 = X_red_1[:len(X_a)]
X_b_red_1 = X_red_1[len(X_a):]

# Approach 2: Diffusion Maps with Diffusion Loss
encoder = build_seq_encoder(input_shape=X_a.shape[1:], filters=8, n_components=2, zero_padding=0, use_bn=False)
tic = time.perf_counter()
loss = DiffusionLoss(X_a, sigma=sigma, steps=steps, alpha=alpha)
encoder.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate=0.005))
indices = np.array(list(range(X_a.shape[0])))
hist_enc = encoder.fit(x=X_a, y=indices, epochs=750, validation_split=0.1, shuffle=False, batch_size=128, verbose=0)
X_a_red_2 = encoder(X_a)
tac = time.perf_counter()
X_b_red_2 = encoder(X_b)
toc = time.perf_counter()

encoder.save(os.path.join(output_dir, 'encoder.keras'))
with h5py.File(os.path.join(output_dir, 'hist_enc.h5'), 'w') as file:
    for key, value in hist_enc.history.items():
        file.create_dataset(key, data=value)

mae_2, mae_2_conf_int = mae(X_b_red_1, X_b_red_2)

# Approach 3: Nyström to extend existing embedding
X_a_red_3 = DM.fit_transform(X_a)
X_b_red_3 = DM.transform(X_b)
mae_3, mae_3_conf_int = mae(X_b_red_1, X_b_red_3)

with h5py.File(os.path.join(output_dir, 'results.h5'), "w") as file:
    # Group for hyperparameters
    group_hyperparameters = file.create_group("hyperparameters")
    group_hyperparameters.create_dataset("n_components", data=n_components)
    group_hyperparameters.create_dataset("q", data=q)
    group_hyperparameters.create_dataset("alpha", data=alpha)
    group_hyperparameters.create_dataset("steps", data=steps)
    group_hyperparameters.create_dataset("sigma", data=sigma)

    # Group for original data
    group_0 = file.create_group("original")
    group_0.create_dataset("X_a", data=X_a, compression='gzip')
    group_0.create_dataset("y_a", data=y_a, compression='gzip')
    group_0.create_dataset("X_b", data=X_b, compression='gzip')
    group_0.create_dataset("y_b", data=y_b, compression='gzip')

    # Group for approach 1
    group_1 = file.create_group("difussion_maps")
    group_1.create_dataset("X_a_red", data=X_a_red_1, compression='gzip')
    group_1.create_dataset("X_b_red", data=X_b_red_1, compression='gzip')

    # Group for approach 2
    group_2 = file.create_group("deep_diffusion_maps")
    group_2.create_dataset("X_a_red", data=X_a_red_2, compression='gzip')
    group_2.create_dataset("X_b_red", data=X_b_red_2, compression='gzip')
    group_2.create_dataset("mae", data=mae_2)
    group_2.create_dataset("mae_conf_int", data=mae_2_conf_int)

    # Group for approach 3
    group_3 = file.create_group("nystrom")
    group_3.create_dataset("X_a_red", data=X_a_red_3, compression='gzip')
    group_3.create_dataset("X_b_red", data=X_b_red_3, compression='gzip')
    group_3.create_dataset("mae", data=mae_3)
    group_3.create_dataset("mae_conf_int", data=mae_3_conf_int)
