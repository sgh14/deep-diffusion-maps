import os
import numpy as np
from scipy.spatial.distance import pdist
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from DiffusionMaps import DiffusionMaps


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


def plot_loglikelihood(X, q_vals, alpha_vals, steps_vals, output_dir='', max_components=None):
    max_components = max_components if max_components else (X.shape[-1] - 1)
    n_components_vals = np.arange(1, max_components + 1)
    fig, axes = plt.subplots(len(alpha_vals), 1, figsize=(6, 1+3*len(alpha_vals)), sharex=True, sharey=True, squeeze=False)
    for ax, alpha in zip(axes.flatten(), alpha_vals):
        for j, q in enumerate(q_vals):
            DM = DiffusionMaps(get_sigma(X, q), 1, 1, alpha)
            _ = DM.fit_transform(X)
            for k, steps in enumerate(steps_vals):
                eigenvalues = DM.lambdas[1:]**steps
                l_vals = log_likelihood_curve(eigenvalues, max_components)
                normalized_l_vals = (l_vals-np.min(l_vals))/(np.max(l_vals)-np.min(l_vals))

                ax.plot(n_components_vals, normalized_l_vals, color=colors[k], linestyle=linestyles[j])
                if len(alpha_vals) > 1:
                    ax.set_title(f'$\\alpha = {alpha}$')
                ax.set_ylabel('Normalized log-likelihood')

    ax.set_xlabel('$d$')
    # ax.set_xticks(n_components_vals)
    # Create custom legends
    q_legend = {f'${q}$': Line2D([0], [0], linewidth=2, color='gray', linestyle=linestyles[i]) for i, q in enumerate(q_vals)}
    steps_legend = {f'${steps}$': Line2D([0], [0], linewidth=2, color=colors[i]) for i, steps in enumerate(steps_vals)}
    scale_factor = 2.5/len(alpha_vals)
    if len(q_vals) > 1:
        fig.legend(q_legend.values(), q_legend.keys(), title="Value of $q$", loc='lower center', bbox_to_anchor=(0.5, -0.05*scale_factor), ncol=6, handletextpad=0.3, columnspacing=0.3)
    if len(steps_vals) > 1:
        fig.legend(steps_legend.values(), steps_legend.keys(), title="Value of $t$", loc='lower center', bbox_to_anchor=(0.5, -(0.05*scale_factor + 0.08)), ncol=7, handletextpad=0.3, columnspacing=0.3, handlelength=2.5)
    for format in ('.pdf',):# '.png', '.svg'):
        fig.savefig(os.path.join(output_dir, 'l_values' + format))

    plt.close(fig)


def plot_eigenvalues(X, q_vals, alpha_vals, steps_vals, output_dir='', max_components=None, log_scale=False):
    max_components = max_components if max_components else (X.shape[-1] - 1)
    x = np.arange(1, max_components + 1)
    fig, axes = plt.subplots(len(alpha_vals), 1, figsize=(6, 1+3*len(alpha_vals)), sharex=True, sharey=True, squeeze=False)
    for ax, alpha in zip(axes.flatten(), alpha_vals):
        for j, q in enumerate(q_vals):
            DM = DiffusionMaps(get_sigma(X, q), 1, 1, alpha)
            _ = DM.fit_transform(X)
            for k, steps in enumerate(steps_vals):
                eigenvalues = DM.lambdas[1:]**steps
                y = np.log(eigenvalues[:max_components]) if log_scale else eigenvalues[:max_components]
                ax.plot(x, y, color=colors[k], linestyle=linestyles[j])
                if len(alpha_vals) > 1:
                    ax.set_title(f'$\\alpha = {alpha}$')
                ax.set_ylabel('$\\log(\\lambda^t)$' if log_scale else '$\\lambda^t$')

    ax.set_xlabel('$d$')
    # ax.set_xticks(x)
    # Create custom legends
    q_legend = {f'${q}$': Line2D([0], [0], linewidth=2, color='gray', linestyle=linestyles[i]) for i, q in enumerate(q_vals)}
    steps_legend = {f'${steps}$': Line2D([0], [0], linewidth=2, color=colors[i]) for i, steps in enumerate(steps_vals)}
    scale_factor = 3/len(alpha_vals)
    if len(q_vals) > 1:
        fig.legend(q_legend.values(), q_legend.keys(), title="Value of $q$", loc='lower center', bbox_to_anchor=(0.5, -0.05*scale_factor), ncol=6, handletextpad=0.3, columnspacing=0.3)
    if len(steps_vals) > 1:
        fig.legend(steps_legend.values(), steps_legend.keys(), title="Value of $t$", loc='lower center', bbox_to_anchor=(0.5, -(0.05*scale_factor + 0.08)), ncol=7, handletextpad=0.3, columnspacing=0.3, handlelength=2.5)
    for format in ('.pdf',):# '.png', '.svg'):
        fig.savefig(os.path.join(output_dir, 'eigenvalues' + format))
    
    plt.close(fig)


def percentile_curve(distances):
    q_vals = np.linspace(0, 1, 201)
    percentiles = np.quantile(distances, q_vals)

    return q_vals, percentiles


def get_elements_above_diagonal(A):
    # Get the indices of the upper triangle (excluding the diagonal)
    rows, cols = np.triu_indices(A.shape[0], k=1)

    # Extract the elements above the diagonal
    upper_triangle_vector = A[rows, cols]

    return upper_triangle_vector


def my_colormap1D(x, c1=(0.75, 0, 0.75), c2=(0, 0.75, 0.75)):
    # Calculate the RGB values based on interpolation
    color = np.array(c1) * (1 - x) + np.array(c2) * x

    return color


def plot_distance_percentiles(X, q_vals, alpha_vals, steps_vals, output_dir=''):
    D_euc = pdist(X, metric='euclidean')
    x_euc, y_euc = percentile_curve(D_euc/np.max(D_euc))
    fig, axes = plt.subplots(
        len(alpha_vals),
        len(steps_vals),
        figsize=(3*len(steps_vals), 3*len(alpha_vals)),
        sharex=True, sharey=True, squeeze=False
    )
    for i, alpha in enumerate(alpha_vals):
        for k, steps in enumerate(steps_vals):
            ax = axes[i, k]
            ax.plot(x_euc, y_euc, color='black', linestyle='--')
            for j, q in enumerate(q_vals):
                DM = DiffusionMaps(get_sigma(X, q), 1, 1, alpha)
                _ = DM.fit_transform(X)
                P = DM.transition_probabilities(DM.W)
                P_steps = np.linalg.matrix_power(P, steps)
                D_diff = DM.diffusion_distances(P_steps, DM.pi)
                x_diff, y_diff = percentile_curve(get_elements_above_diagonal(D_diff/np.max(D_diff)))
                ax.plot(x_diff, y_diff, color=my_colormap1D(j/len(q_vals)), linestyle='-')
                if len(alpha_vals) > 1:
                    ax.set_title(f'$\\alpha = {alpha}$, $t = {steps}$')
                
                if k == 0:
                    ax.set_ylabel('Normalized diffusion distance')
                if i == len(alpha_vals) - 1:
                    ax.set_xlabel('Percentile')

    # ax.set_xticks(x)
    # Create custom legends
    q_legend = {f'${q}$': Line2D([0], [0], linewidth=2, color=my_colormap1D(i/len(q_vals)), linestyle='-') for i, q in enumerate(q_vals)}
    if len(q_vals) > 1:
        fig.legend(q_legend.values(), q_legend.keys(), title="Value of $q$", loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=6, handletextpad=0.3, columnspacing=0.3)
    for format in ('.pdf',):# '.png', '.svg'):
        fig.savefig(os.path.join(output_dir, 'distances' + format))
    
    plt.close(fig)


# def plot_distance_percentiles(X, q_vals, alpha_vals, steps_vals, output_dir=''):
#     D_euc = pdist(X, metric='euclidean')
#     x_euc, y_euc = percentile_curve(D_euc/np.max(D_euc))
#     fig, axes = plt.subplots(len(alpha_vals), 1, figsize=(6, 1+3*len(alpha_vals)), sharex=True, sharey=True, squeeze=False)
#     for ax, alpha in zip(axes.flatten(), alpha_vals):
#         ax2 = ax.twinx()
#         ax2.plot(x_euc, y_euc, color='red', linestyle='-')
#         ax2.set_ylabel('Normalized euclidean distance', color='red')
#         for j, q in enumerate(q_vals):
#             DM = DiffusionMaps(get_sigma(X, q), 1, 1, alpha)
#             _ = DM.fit_transform(X)
#             P = DM.transition_probabilities(DM.W)
#             for k, steps in enumerate(steps_vals):
#                 P_steps = np.linalg.matrix_power(P, steps)
#                 D_diff = DM.diffusion_distances(P_steps, DM.pi)
#                 x_diff, y_diff = percentile_curve(get_elements_above_diagonal(D_diff/np.max(D_diff)))
#                 ax.plot(x_diff, y_diff, color=colors[k], linestyle=linestyles[j])
#                 if len(alpha_vals) > 1:
#                     ax.set_title(f'$\\alpha = {alpha}$')
#                 ax.set_ylabel('Normalized diffusion distance')

#     ax.set_xlabel('Percentile')
#     # ax.set_xticks(x)
#     # Create custom legends
#     q_legend = {f'${q}$': Line2D([0], [0], linewidth=2, color='gray', linestyle=linestyles[i]) for i, q in enumerate(q_vals)}
#     steps_legend = {f'${steps}$': Line2D([0], [0], linewidth=2, color=colors[i]) for i, steps in enumerate(steps_vals)}
#     scale_factor = 3/len(alpha_vals)
#     if len(q_vals) > 1:
#         fig.legend(q_legend.values(), q_legend.keys(), title="Value of $q$", loc='lower center', bbox_to_anchor=(0.5, -0.05*scale_factor), ncol=6, handletextpad=0.3, columnspacing=0.3)
#     if len(steps_vals) > 1:
#         fig.legend(steps_legend.values(), steps_legend.keys(), title="Value of $t$", loc='lower center', bbox_to_anchor=(0.5, -(0.05*scale_factor + 0.08)), ncol=7, handletextpad=0.3, columnspacing=0.3, handlelength=2.5)
#     for format in ('.pdf',):# '.png', '.svg'):
#         fig.savefig(os.path.join(output_dir, 'distances' + format))
    
#     plt.close(fig)