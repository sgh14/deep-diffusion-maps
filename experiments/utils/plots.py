import os
import numpy as np
from typing import Tuple, List, Dict
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

plt.style.use('experiments/science.mplstyle')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
linestyles = ['-', '--', '-.', ':', (5, (10, 3)), (0, (3, 1, 1, 1, 1, 1))]


def normalize(x: np.ndarray) -> np.ndarray:
    """
    Normalize an array to the range [0, 1] using min-max normalization.
    
    Parameters:
        x (np.ndarray): Input array to normalize.
        
    Returns:
        np.ndarray: Normalized array with values scaled to [0, 1].
    """
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def colormap1D(
        x: float,
        c1: Tuple[float, float, float]=(0.75, 0, 0.75),
        c2: Tuple[float, float, float]=(0, 0.75, 0.75)
) -> np.ndarray:
    """
    Create a color by linearly interpolating between two colors based on a value x.
    
    Parameters:
        x (float): Value between 0 and 1 that determines the interpolation amount.
        c1 (tuple): RGB values for the color at x=0 (default: dark magenta).
        c2 (tuple): RGB values for the color at x=1 (default: dark cyan).
        
    Returns:
        np.ndarray: Interpolated RGB color values.
    
    Example:
        >>> colormap1D(0)
        array([0.75, 0.  , 0.75])  # Returns c1 (dark magenta)
        >>> colormap1D(1)
        array([0.  , 0.75, 0.75])  # Returns c2 (dark cyan)
        >>> colormap1D(0.5)
        array([0.375, 0.375, 0.75])  # Returns the midpoint color
    """
    # Calculate the RGB values based on linear interpolation
    color = np.array(c1) * (1 - x) + np.array(c2) * x

    return color


def colormap2D(x: float, y: float) -> np.ndarray:
    """
    Create a color by bilinearly interpolating between four colors based on x and y values.
    
    This function creates a 2D color map with different colors at the four corners of
    a unit square, and interpolates between them based on the x and y coordinates.
    
    Parameters:
        x (float): Value between 0 and 1 representing horizontal position.
        y (float): Value between 0 and 1 representing vertical position.
        
    Returns:
        np.ndarray: Interpolated RGB color values.
        
    Corner colors:
        - Bottom left (x=0, y=0): Dark magenta (0.5, 0, 0.5)
        - Bottom right (x=1, y=0): Dark cyan (0, 0.5, 0.5)
        - Top left (x=0, y=1): Magenta (1, 0, 1)
        - Top right (x=1, y=1): Cyan (0, 1, 1)
    """
    # Define colors in RGB
    bottom_left = (0.5, 0, 0.5)  # dark magenta
    bottom_right = (0, 0.5, 0.5)  # dark cyan
    top_left = (1, 0, 1)  # magenta
    top_right = (0, 1, 1)  # cyan

    # First interpolate along x-axis for both top and bottom rows
    top_color = np.array(top_left) * (1 - x) + np.array(top_right) * x
    bottom_color = np.array(bottom_left) * (1 - x) + np.array(bottom_right) * x
    
    # Then interpolate along y-axis between top and bottom results
    return bottom_color * (1 - y) + top_color * y


def get_max_range(X: np.ndarray) -> float:
    """
    Calculate the maximum range of values across all dimensions of a dataset.

    Parameters:
        X (np.ndarray): Input dataset with shape (n_samples, n_features).

    Returns:
        float: The maximum range of values across all dimensions.
    """
    x_max = np.max(X, axis=0)
    x_min = np.min(X, axis=0)
    x_range = x_max - x_min
    max_range = np.max(x_range)

    return max_range


def set_equal_ranges(ax: plt.Axes, max_range: float) -> plt.Axes:
    """
    Adjust the axis limits to have equal ranges for both x and y axes.

    Parameters:
        ax (plt.Axes): Matplotlib Axes object to adjust.
        max_range (float): Maximum range to set for both axes.

    Returns:
        plt.Axes: The adjusted Matplotlib Axes object.
    """
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


def plot_eigenvalues_and_log_likelihood(
    eigenvalues_curves: List[np.ndarray],
    log_likelihood_curves: List[np.ndarray],
    labels: List[str],
    output_dir: str
) -> None:
    """
    Plot eigenvalue decay and corresponding normalized log-likelihood curves.

    Parameters:
        eigenvalues_curves (List[np.ndarray]): List of eigenvalue sequences.
        log_likelihood_curves (List[np.ndarray]): List of log-likelihood sequences.
        labels (List[str]): Labels for each curve.
        output_dir (str): Directory to save the plots.

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    max_components = min([len(l_curve) for l_curve in log_likelihood_curves])
    n_components_vals = np.arange(1, max_components + 1)
    for i in range(len(eigenvalues_curves)):
        e_curve = eigenvalues_curves[i][:max_components]
        l_curve = log_likelihood_curves[i][:max_components]
        normalized_l_curve = l_curve / np.max(l_curve)
        ax1.plot(
            n_components_vals,
            e_curve,
            color=colors[0],
            linestyle=linestyles[i]
        )
        ax2.plot(
            n_components_vals,
            normalized_l_curve,
            color=colors[1],
            linestyle=linestyles[i]
        )

    ax1.set_xlabel('$d$')
    ax1.set_ylabel('$\\lambda^t_d$', color=colors[0])
    ax2.set_ylabel('Normalized log-likelihood', color=colors[1])
    ax1.tick_params(axis='y', labelcolor=colors[0])
    ax2.tick_params(axis='y', labelcolor=colors[1])
    if len(eigenvalues_curves) > 1:
        legend = {
            label: Line2D([0], [0], linewidth=2, color='gray', linestyle=linestyles[i])
            for i, label in enumerate(labels)
        }
        fig.legend(
            legend.values(),
            legend.keys(),
            loc='lower center',
            bbox_to_anchor=(0.5, -0.1),
            ncol=4,
            handletextpad=0.3,
            columnspacing=0.3
        )

    ax1.ticklabel_format(axis='y', style='sci', scilimits=(-1, 1), useMathText=True)
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(-1, 1), useMathText=True)

    for format in ('.pdf',):# '.png', '.svg'):
        fig.savefig(os.path.join(output_dir, 'eigenvalues-log_likelihood' + format))

    plt.close(fig)


def plot_history(
    history: Dict[str, List[float]],
    output_dir: str,
    filename: str,
    logy: bool = False,
    logx: bool = False
) -> None:
    """
    Plot training/validation metrics over epochs.

    Parameters:
        history (Dict[str, List[float]]): Dictionary containing keys like 'loss' and 'val_loss'.
        output_dir (str): Directory to save the plots.
        filename (str): Base name for the saved file.
        logy (bool): Use log-scale on the y-axis (default: False).
        logx (bool): Use log-scale on the x-axis (default: False).

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)

    metric_keys = [k for k in history if not k.startswith('val_')]
    for key in metric_keys:
        y_train = np.array(history[key])
        y_val = np.array(history['val_' + key])
        fig, ax = plt.subplots()

        if logx and logy:
            ax.loglog(y_train, label='Training')
            ax.loglog(y_val, label='Validation')
        elif logy:
            ax.semilogy(y_train, label='Training')
            ax.semilogy(y_val, label='Validation')
        else:
            ax.plot(y_train, label='Training')
            ax.plot(y_val, label='Validation')
            ax.ticklabel_format(axis='both', style='sci', scilimits=(-1, 1), useMathText=True)

        ax.set_xlabel('Epoch')
        ax.set_ylabel(key.capitalize())
        ax.legend()
        for format in ('.pdf',):# '.png', '.svg'):
            fig.savefig(os.path.join(output_dir, filename + '-' + key + format))
        
        plt.close(fig)


def plot_mre_by_decile(
    decile_distances: np.ndarray,
    mre_by_decile: List[np.ndarray],
    labels: List[str],
    output_dir: str,
    filename: str
) -> None:
    """
    Plot mean relative error (MRE) across distance deciles.

    Parameters:
        decile_distances (np.ndarray): Distance thresholds per decile.
        mre_by_decile (List[np.ndarray]): MRE values for each method or model.
        labels (List[str]): Labels for each method or model.
        output_dir (str): Directory to save the plot.
        filename (str): Output file name (no extension).

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    decile_points = list(range(0, 101, 10))
    x_pos = np.arange(len(decile_points) - 1) + 0.5
    fig, ax1 = plt.subplots()
    ax2 = ax1.twiny()
    for i in range(len(mre_by_decile)):
        ax1.bar(
            x_pos,
            mre_by_decile[i],
            width=1.0,
            edgecolor=colors[i],
            facecolor='white',
            label=labels[i],
            linewidth=2.0
        )

    ax1.set_xticks(np.arange(len(decile_points)))
    ax1.set_xticklabels(decile_points)
    ax1.set_xlim(0, len(decile_points) - 1)

    ax2.set_xticks(np.arange(len(decile_distances)))
    ax2.set_xticklabels([f"${d:.1e}$" for d in decile_distances])
    ax2.set_xlim(0, len(decile_distances) - 1)
    ax2.tick_params(axis='x', labelrotation=90)

    ax1.set_xlabel('Decile')
    ax1.set_ylabel('MRE')
    ax2.set_xlabel('Euclidean distance in the embedding')
    ax1.legend()
    for format in ('.pdf',):# '.png', '.svg'):
        fig.savefig(os.path.join(output_dir, filename + format))