import os
import h5py
import numpy as np
from matplotlib import pyplot as plt

from experiments.utils.experiments import load_config
from experiments.utils.plots import plot_eigenvalues_and_log_likelihood, plot_history, plot_mre_by_decile, get_max_range, set_equal_ranges

plt.style.use('experiments/science.mplstyle')


def plot_original(X, y, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(3, 3), subplot_kw={"projection": "3d"})
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, alpha=1)
    ax.set_xlim([-13, 13])
    ax.set_ylim([-3, 23])
    ax.set_zlim([-13, 13])
    ax.view_init(15, -72)
    # ax.dist = 12
    ax.grid(False)
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
                ax.scatter(X[:, dim1], X[:, dim2], c=y)
                # Remove the ticks
                ax.set_xticks([])
                ax.set_yticks([])
                # Remove the tick labels
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax = set_equal_ranges(ax, max_range)

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
                plot_eigenvalues_and_log_likelihood([eigenvalues], [log_likelihood], ['Swiss Roll'], output_dir)
        

if __name__ == "__main__":
    main()