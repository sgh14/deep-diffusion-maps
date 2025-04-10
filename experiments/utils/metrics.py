import numpy as np
from typing import Tuple, List, Dict, Any
from scipy.spatial.distance import pdist


def get_deciles(x: np.ndarray) -> np.ndarray:
    """
    Compute decile thresholds for a numeric array.

    Parameters:
        x (np.ndarray): Input array.

    Returns:
        np.ndarray: Array of decile thresholds from 0% to 100% in 10% increments.
    """
    deciles = np.percentile(x, np.arange(0, 101, 10))
    
    return deciles


def get_mean_value_by_decile(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, List[float]]:
    """
    Compute the mean of values in `y` corresponding to decile bins of `x`.

    Parameters:
        x (np.ndarray): Reference values for binning.
        y (np.ndarray): Values to compute the mean of in each bin.

    Returns:
        Tuple[np.ndarray, List[float]]: Decile thresholds and list of mean values per bin.
    """
    deciles = get_deciles(x)
    mean_values = []
    for i in range(len(deciles) - 1):
        mask = (x >= deciles[i]) & (x < deciles[i + 1])
        mean_values.append(np.mean(y[mask]))
    
    return deciles, mean_values


def bootstrap_ci(x: np.ndarray, conf_level: float = 0.95, n_bootstraps: int = 1000) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for the mean of `x`.

    Parameters:
        x (np.ndarray): Input array of values.
        conf_level (float): Confidence level for the interval (default is 0.95).
        n_bootstraps (int): Number of bootstrap samples (default is 1000).

    Returns:
        Tuple[float, float]: Lower and upper bounds of the confidence interval.
    """
    bootstraps = []
    for _ in range(n_bootstraps):
        sample_indices = np.random.choice(len(x), len(x), replace=True)
        sample_mean = np.mean(x[sample_indices])
        bootstraps.append(sample_mean)

    lower_bound = np.percentile(bootstraps, (1 - conf_level) / 2 * 100)
    upper_bound = np.percentile(bootstraps, (1 + conf_level) / 2 * 100)

    return lower_bound, upper_bound


def mae(x_true: np.ndarray, x_pred: np.ndarray, conf_level: float = 0.95) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """
    Compute Mean Absolute Error (MAE) and confidence interval.

    Parameters:
        x_true (np.ndarray): Ground truth values.
        x_pred (np.ndarray): Predicted values.
        conf_level (float): Confidence level for the interval (default is 0.95).

    Returns:
        Tuple[np.ndarray, float, Tuple[float, float]]:
            - Array of absolute errors
            - Mean absolute error
            - Confidence interval of the mean
    """
    errors = np.abs(x_true - x_pred)
    mean = np.mean(errors)
    conf_int = bootstrap_ci(errors, conf_level)

    return errors, mean, conf_int


def mre(x_true: np.ndarray, x_pred: np.ndarray, conf_level: float = 0.95) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """
    Compute Mean Relative Error (MRE) and confidence interval.

    Parameters:
        x_true (np.ndarray): Ground truth values.
        x_pred (np.ndarray): Predicted values.
        conf_level (float): Confidence level for the interval (default is 0.95).

    Returns:
        Tuple[np.ndarray, float, Tuple[float, float]]:
            - Array of relative errors
            - Mean relative error
            - Confidence interval of the mean
    """
    errors = np.abs(x_true - x_pred) / np.abs(x_true)
    mean = np.mean(errors)
    conf_int = bootstrap_ci(errors, conf_level)

    return errors, mean, conf_int


def distances_errors(x_true: np.ndarray, x_pred: np.ndarray, conf_level: float = 0.95) -> Dict[str, Any]:
    """
    Compute MAE and MRE for pairwise distances between samples,
    including their values binned by distance deciles.

    Parameters:
        x_true (np.ndarray): Ground truth samples (2D).
        x_pred (np.ndarray): Predicted samples (2D).
        conf_level (float): Confidence level for MAE and MRE intervals (default is 0.95).

    Returns:
        dict: {
            'decile_distances': np.ndarray of distance deciles,
            'mae_by_decile': List[float],
            'mae_mean': float,
            'mae_conf_int': Tuple[float, float],
            'mre_by_decile': List[float],
            'mre_mean': float,
            'mre_conf_int': Tuple[float, float]
        }
    """
    dists_true = pdist(x_true.reshape((x_true.shape[0], -1)), metric='euclidean')
    dists_pred = pdist(x_pred.reshape((x_pred.shape[0], -1)), metric='euclidean')
    mae_vals, mae_mean, mae_conf_int = mae(dists_true, dists_pred, conf_level)
    mre_vals, mre_mean, mre_conf_int = mre(dists_true, dists_pred, conf_level)
    decile_distances, mae_by_decile = get_mean_value_by_decile(dists_true, mae_vals)
    _, mre_by_decile = get_mean_value_by_decile(dists_true, mre_vals)

    return {
        'decile_distances': decile_distances,
        'mae_by_decile': mae_by_decile,
        'mae_mean': mae_mean,
        'mae_conf_int': mae_conf_int,
        'mre_by_decile': mre_by_decile,
        'mre_mean': mre_mean,
        'mre_conf_int': mre_conf_int
    }