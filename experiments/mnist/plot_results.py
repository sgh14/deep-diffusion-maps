import os
import h5py
import numpy as np
from matplotlib import pyplot as plt
import argparse
import yaml

from experiments.aux_functions import plot_eigenvalues_and_log_likelihood, plot_history

plt.style.use('experiments/science.mplstyle')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


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


# Function to sample 2 images per class from the dataset
def sample_images_per_class(X, y, images_per_class=2):
    selected_images = []
    selected_labels = []
    
    n_classes = len(np.unique(y))
    for class_label in range(n_classes):
        class_indices = np.where(y == class_label)[0]
        selected_indices = class_indices[:images_per_class]
        selected_images.extend(X[selected_indices])
        selected_labels.extend(y[selected_indices])
    
    return np.array(selected_images), np.array(selected_labels)


def plot_images(axes, X, y=[]):
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(X[i], cmap='gray')
        if len(y) > 0:
            ax.set_title(y[i])

        ax.axis('off')
    
    return axes


def plot_original(
    X,
    y,
    output_dir,
    filename,
    images_per_class=2,
    grid_shape=(3, 4)
):
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(
        grid_shape[0], grid_shape[1],
        figsize=(3, 3),
        gridspec_kw={'wspace': 0.2, 'hspace': 0.2}
    )
    X, y = sample_images_per_class(X, y, images_per_class)
    axes = plot_images(axes, X, y)
    fig.tight_layout()
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
                ax.scatter(X[:, dim1], X[:, dim2], c=[colors[i] for i in y])
                # # Remove the ticks
                # ax.set_xticks([])
                # ax.set_yticks([])
                # # Remove the tick labels
                # ax.set_xticklabels([])
                # ax.set_yticklabels([])
                # ax.set_xlabel(r'$\Psi_1$')
                # ax.set_ylabel(r'$\Psi_2$')
                ax = set_equal_ranges(ax, max_range) # ax.set_box_aspect(1)
                # Create a list of handles and labels for the legend
                unique_y = np.unique(y)
                handles = [plt.Line2D([0], [0], linewidth=2, marker='o', color='w', markerfacecolor=colors[val], markersize=10) for val in unique_y]
                labels = [str(val) for val in unique_y]  # Adjust labels based on your case
                fig.legend(handles, labels, loc='lower center', ncol=len(unique_y), handletextpad=0.2, columnspacing=0.2, bbox_to_anchor=(0.5, -0.12))

                for format in ('.pdf',):# '.png', '.svg'):
                    fig.savefig(os.path.join(output_dir, filename + f'_dims_{dim1+1}_{dim2+1}' + format))
                
                plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(3, 3), constrained_layout=True)
        ax.scatter(X[:, 0], np.zeros(X.shape[0]), c=y)
        # # Remove the ticks
        # ax.set_xticks([])
        # ax.set_yticks([])
        # # Remove the tick labels
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        # ax.set_xlabel(r'$\Psi_1$')
        # ax.set_ylabel(r'$\Psi_2$')
        ax = set_equal_ranges(ax, max_range) # ax.set_box_aspect(1)
        # Create a list of handles and labels for the legend
        unique_y = np.unique(y)
        handles = [plt.Line2D([0], [0], linewidth=2, marker='o', color='w', markerfacecolor=colors[val], markersize=10) for val in unique_y]
        labels = [str(val) for val in unique_y]  # Adjust labels based on your case
        fig.legend(handles, labels, loc='lower center', ncol=len(unique_y), handletextpad=0.2, columnspacing=0.2, bbox_to_anchor=(0.5, -0.12))

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

plot_history(history, output_dir, 'hist_enc', logy=True, logx=True)

# Load data from the HDF5 file
with h5py.File(os.path.join(output_dir, 'results.h5'), "r") as file:
    # Load original data
    group_0 = file["original"]
    X_a = np.array(group_0["X_a"][:])
    y_a = np.array(group_0["y_a"][:])
    X_b = np.array(group_0["X_b"][:])
    y_b = np.array(group_0["y_b"][:])

    # Load data for approach 1
    group_1 = file["difussion_maps"]
    X_a_red_1 = np.array(group_1["X_a_red"][:])
    X_b_red_1 = np.array(group_1["X_b_red"][:])

    # Load data for approach 2
    group_2 = file["deep_diffusion_maps"]
    X_a_red_2 = np.array(group_2["X_a_red"][:])
    X_b_red_2 = np.array(group_2["X_b_red"][:])

    # Load data for approach 3
    group_3 = file["nystrom"]
    X_a_red_3 = np.array(group_3["X_a_red"][:])
    X_b_red_3 = np.array(group_3["X_b_red"][:])
    eigenvalues = np.array(group_3["eigenvalues"][:])
    log_likelihood = np.array(group_3["log_likelihood"][:])
                 
plot_original(X_a, y_a, output_dir, 'orig_a', images_per_class=2, grid_shape=(3, 4))
plot_original(X_b, y_b, output_dir, 'orig_b', images_per_class=2, grid_shape=(3, 4))
plot_projection(X_a_red_1, y_a, output_dir, 'red_a_dm')
plot_projection(X_a_red_2, y_a, output_dir, 'red_a_ddm')
plot_projection(X_a_red_3, y_a, output_dir, 'red_a_nys')
plot_projection(X_b_red_1, y_b, output_dir, 'red_b_dm')
plot_projection(X_b_red_2, y_b, output_dir, 'red_b_ddm')
plot_projection(X_b_red_3, y_b, output_dir, 'red_b_nys')
plot_eigenvalues_and_log_likelihood([eigenvalues], [log_likelihood], ['MNIST'], output_dir)
