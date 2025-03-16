import os
import h5py
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import argparse
import yaml

from experiments.aux_functions import plot_eigenvalues_and_log_likelihood, plot_history, plot_mre_by_decile

plt.style.use('experiments/science.mplstyle')


def get_max_range(X):
    x_max = np.max(X, axis=0)
    x_min = np.min(X, axis=0)
    x_range = x_max - x_min
    max_range = np.max(x_range)

    return max_range


def set_equal_ranges(ax, max_range):
    # Get the current axis limits
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # Set new limits with the same range for both axes
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2

    max_range = max_range * 1.05

    ax.set_xlim(x_center - max_range / 2, x_center + max_range / 2)
    ax.set_ylim(y_center - max_range / 2, y_center + max_range / 2)

    # Set equal aspect ratio
    ax.set_aspect('equal', adjustable='box')

    return ax


def plot_original(X, y, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(3, 3), subplot_kw={"projection": "3d"})
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, alpha=1)
    # ax.set_xlabel('$x_1$')
    ax.set_xlim([-1, 1])
    # ax.set_ylabel('$x_2$')
    ax.set_ylim([0, 2])
    # ax.set_zlabel('$x_3$')
    ax.set_zlim([-2, 2])
    ax.view_init(15, -72)
    # ax.dist = 12
    ax.grid(False)
    fig.tight_layout()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.zaxis.set_major_locator(ticker.MultipleLocator(1))

    for format in ('.pdf',):# '.png', '.svg'):
        fig.savefig(os.path.join(output_dir, filename + format))

    plt.close(fig)


def plot_projection(X, y, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    max_range = get_max_range(X)
    ndims = X.shape[-1]
    if ndims > 1:
        for dim1 in range(0, ndims):
            for dim2 in range(dim1 + 1, ndims):
                fig, ax = plt.subplots(figsize=(3, 3), constrained_layout=True)
                ax.scatter(X[:, dim1], X[:, dim2], c=y)
                # Remove the ticks
                ax.set_xticks([])
                ax.set_yticks([])
                # Remove the tick labels
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                # ax.set_xlabel(r'$\Psi_1$')
                # ax.set_ylabel(r'$\Psi_2$')
                ax = set_equal_ranges(ax, max_range) # ax.set_box_aspect(1)

                for format in ('.pdf',):# '.png', '.svg'):
                    fig.savefig(os.path.join(output_dir, filename + f'_dims_{dim1+1}_{dim2+1}' + format))
                
                plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(3, 3), constrained_layout=True)
        ax.scatter(X[:, 0], np.zeros(X.shape[0]), c=y)
        # Remove the ticks
        ax.set_xticks([])
        ax.set_yticks([])
        # Remove the tick labels
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # ax.set_xlabel(r'$\Psi_1$')
        # ax.set_ylabel(r'$\Psi_2$')
        ax = set_equal_ranges(ax, max_range) # ax.set_box_aspect(1)

        for format in ('.pdf',):# '.png', '.svg'):
            fig.savefig(os.path.join(output_dir, filename + f'_dims_{1}' + format))
        
        plt.close(fig)


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

with h5py.File(os.path.join(output_dir, 'hist_enc.h5'), 'r') as file:
    history = {key: np.array(file[key]) for key in file.keys()}

plot_history(history, output_dir, 'hist_enc', logy=True, logx=False)

with h5py.File(os.path.join(output_dir, 'results.h5'), "r") as file:
    for subset in ('train', 'test'):
        X = np.array(file["original"][subset]["X"][:])
        y = np.array(file["original"][subset]["y"][:])
        plot_original(X, y, output_dir, f'original_{subset}')

        mre_by_decile = []
        for method in ('diffusion_maps', 'deep_diffusion_maps', 'nystrom'):
            X_red = np.array(file[method][subset]["X_red"][:])
            plot_projection(X_red, y, output_dir, f'projection_{method}_{subset}')
            if method != 'diffusion_maps':
                decile_distances = np.array(file[method][subset]["decile_distances"][:])
                mre_by_decile.append(np.array(file[method][subset]["mre_by_decile"][:]))
            
        plot_mre_by_decile(decile_distances, mre_by_decile, ['DDM', 'Nystrom'], output_dir, f'mre_dist_{subset}')
        
        if subset == 'train':
            P_color = np.array(file['nystrom'][subset]["P_color"][:])
            D_color = np.array(file['nystrom'][subset]["D_color"][:])
            eigenvalues = np.array(file['nystrom'][subset]["eigenvalues"][:])
            log_likelihood = np.array(file['nystrom'][subset]["log_likelihood"][:])

            plot_original(X, P_color, output_dir, f'probs_{subset}')
            plot_original(X, D_color, output_dir, f'dists_{subset}')
            plot_eigenvalues_and_log_likelihood([eigenvalues], [log_likelihood], ['S Curve'], output_dir)
