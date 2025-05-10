import os
import h5py
import numpy as np
from matplotlib import pyplot as plt

from experiments.utils.experiments import load_config
from experiments.utils.plots import plot_eigenvalues_and_log_likelihood, plot_history, plot_mre_by_decile, get_max_range, set_equal_ranges

plt.style.use('experiments/science.mplstyle')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


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
        ax.imshow(X[i].squeeze(), cmap='gray')
        if len(y) > 0:
            # ax.set_title(y[i])
            ax.text(0.5, -0.1, str(y[i]), horizontalalignment='center', verticalalignment='top', transform=ax.transAxes)

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
                # Remove the ticks
                ax.set_xticks([])
                ax.set_yticks([])
                # Remove the tick labels
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax = set_equal_ranges(ax, max_range)
                # Create a list of handles and labels for the legend
                unique_y = np.unique(y)
                handles = [plt.Line2D([0], [0], linewidth=2, marker='o', color='w', markerfacecolor=colors[val], markersize=10) for val in unique_y]
                labels = [str(val) for val in unique_y]
                fig.legend(handles, labels, loc='lower center', ncol=len(unique_y), handletextpad=0.2, columnspacing=0.2, bbox_to_anchor=(0.5, -0.12))

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
        ax = set_equal_ranges(ax, max_range)
        # Create a list of handles and labels for the legend
        unique_y = np.unique(y)
        handles = [plt.Line2D([0], [0], linewidth=2, marker='o', color='w', markerfacecolor=colors[val], markersize=10) for val in unique_y]
        labels = [str(val) for val in unique_y]
        fig.legend(handles, labels, loc='lower center', ncol=len(unique_y), handletextpad=0.2, columnspacing=0.2, bbox_to_anchor=(0.5, -0.12))

        for format in ('.pdf',):# '.png', '.svg'):
            fig.savefig(os.path.join(output_dir, filename + f'_dims_{1}' + format))
        
        plt.close(fig)


def main():
    # Load configuration
    config, config_path = load_config()
    hyperparametes = config['diffusion_maps']

    # Create directories for output
    root = config['output_dir']
    experiment = '_'.join([k + '_' + str(v) for k, v in hyperparametes.items()])
    output_dir = os.path.join(root, experiment)

    with h5py.File(os.path.join(output_dir, 'history.h5'), 'r') as file:
        history = {key: np.array(file[key]) for key in file.keys()}

    plot_history(history, output_dir, 'history', logy=True, logx=False)

    with h5py.File(os.path.join(output_dir, 'results.h5'), "r") as file:
        for subset in ('train', 'test'):
            X = np.array(file["original"][subset]["X"][:])
            y = np.array(file["original"][subset]["y"][:])
            plot_original(X, y, output_dir, f'original_{subset}', images_per_class=2, grid_shape=(3, 4))

            mre_by_decile = []
            for method in ('diffusion_maps', 'deep_diffusion_maps', 'nystrom'):
                X_red = np.array(file[method][subset]["X_red"][:])
                plot_projection(X_red, y, output_dir, f'projection_{method}_{subset}')
                if method != 'diffusion_maps':
                    decile_distances = np.array(file[method][subset]["decile_distances"][:])
                    mre_by_decile.append(np.array(file[method][subset]["mre_by_decile"][:]))
                
            plot_mre_by_decile(decile_distances, mre_by_decile, ['DDM', 'Nystrom'], output_dir, f'mre_dist_{subset}')
            
            if subset == 'train':
                eigenvalues = np.array(file['nystrom'][subset]["eigenvalues"][:])
                log_likelihood = np.array(file['nystrom'][subset]["log_likelihood"][:])

                plot_eigenvalues_and_log_likelihood([eigenvalues], [log_likelihood], ['MNIST'], output_dir)


if __name__ == "__main__":
    main()