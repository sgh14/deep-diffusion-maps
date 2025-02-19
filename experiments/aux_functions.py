import os
import numpy as np
from scipy.spatial.distance import pdist
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


plt.style.use('experiments/science.mplstyle')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
linestyles = ['-', '--', '-.', ':', (5, (10, 3)), (0, (3, 1, 1, 1, 1, 1))]


def get_sigma(X, q=0.5):
    X_flat = X.reshape((X.shape[0], -1))
    distances = pdist(X_flat, metric='euclidean')
    sigma = np.quantile(distances, q)
    
    return sigma


def log_gaussian_density(x, mean, var):
    log_p = -0.5*np.log(2*np.pi*var) - (x - mean)**2/(2*var)

    return log_p


def log_likelihood(x, y):
    p, q = len(x), len(y)
    sample_mean_1 = np.mean(x)
    sample_var_1 = np.var(x, ddof=1 if len(x) > 1 else 0)
    sample_mean_2 = np.mean(y)
    sample_var_2 = np.var(y, ddof=1 if len(y) > 1 else 0)
    var = ((p - 1)*sample_var_1 + (q - 1)*sample_var_2)/(p + q - 2)
    l = np.sum(log_gaussian_density(x, sample_mean_1, var))\
        + np.sum(log_gaussian_density(y, sample_mean_2, var))
    
    mean_l = l / (p + q)
    
    return mean_l


def log_likelihood_curve(eigenvalues, max_components=25):
    n_components_vals = np.arange(1, max_components + 1)
    l_vals = np.empty((len(n_components_vals),))
    for i, n_components in enumerate(n_components_vals):
        x = eigenvalues[:n_components]
        y = eigenvalues[n_components:]
        l_vals[i] = log_likelihood(x, y)

    return l_vals


def plot_eigenvalues_and_log_likelihood(
        eigenvalues_curves,
        log_likelihood_curves,
        labels,
        output_dir
):
    os.makedirs(output_dir, exist_ok=True)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    max_components = min([len(l_curve) for l_curve in log_likelihood_curves])
    n_components_vals = np.arange(1, max_components + 1)
    for i in range(len(eigenvalues_curves)):
        ax1.plot(
            n_components_vals,
            eigenvalues_curves[i][:max_components],
            color=colors[0],
            linestyle=linestyles[i]
        )
        ax2.plot(
            n_components_vals,
            log_likelihood_curves[i][:max_components],
            color=colors[1],
            linestyle=linestyles[i]
        )

    ax1.set_xlabel('$d$')
    ax1.set_ylabel('$\\lambda^t_d$', color=colors[0])
    ax1.tick_params(axis='y', labelcolor=colors[0])
    ax2.set_ylabel('log-likelihood', color=colors[1])
    ax2.tick_params(axis='y', labelcolor=colors[1])
    # Create custom legends
    if len(eigenvalues_curves) > 1:
        legend = {label: Line2D([0], [0], linewidth=2, color='gray', linestyle=linestyles[i]) for i, label in enumerate(labels)}
        fig.legend(
            legend.values(),
            legend.keys(),
            loc='lower center',
            bbox_to_anchor=(0.5, -0.05),
            ncol=4,
            handletextpad=0.3,
            columnspacing=0.3
        )

    for format in ('.pdf',):# '.png', '.svg'):
        fig.savefig(os.path.join(output_dir, 'eigenvalues-log_likelihood' + format))

    plt.close(fig)
