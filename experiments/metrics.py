import numpy as np
from scipy.spatial.distance import pdist


def get_deciles(x):
    """
    Calculate the deciles of a sample.
    
    Parameters:
    - x: array-like, sample
    
    Returns:
    - deciles: list, deciles of the sample
    """
    deciles = np.percentile(x, np.arange(0, 101, 10))
    
    return deciles


def get_mean_value_by_decile(x, y):
    """
    Calculate the mean value of y for each decile of x.
    
    Parameters:
    - x: array-like, sample
    - y: array-like, values to average
    
    Returns:
    - deciles: list, deciles of x
    - mean_values: list, mean values of y for each decile of x
    """
    deciles = get_deciles(x)
    mean_values = []
    for i in range(len(deciles) - 1):
        mask = (x >= deciles[i]) & (x < deciles[i + 1])
        mean_values.append(np.mean(y[mask]))
    
    return deciles, mean_values


def bootstrap_ci(x, conf_level=0.95):
    """
    Calculate the confidence interval of a sample using bootstrap.
    
    Parameters:
    - x: array-like, sample
    - conf_level: float, confidence level for the interval

    Returns:
    - conf_int: tuple, confidence interval
    """
    bootstraps = []
    for _ in range(1000):
        sample_indices = np.random.choice(len(x), len(x), replace=True)
        sample_mean = np.mean(x[sample_indices])
        bootstraps.append(sample_mean)

    lower_bound = np.percentile(bootstraps, (1 - conf_level) / 2 * 100)
    upper_bound = np.percentile(bootstraps, (1 + conf_level) / 2 * 100)
    conf_int = (lower_bound, upper_bound)

    return conf_int


def mae(x_true, x_pred, conf_level=0.95):
    """
    Mean Absolute Error (MAE) with confidence interval calculation.
    
    Parameters:
    - x_true: array-like, true values
    - x_pred: array-like, predicted values
    - conf_level: float, confidence level for the interval
    
    Returns:
    - mean: float, the MAE
    - conf_int: tuple, confidence interval for MAE
    """
    errors = np.abs(x_true - x_pred)
    mean = np.mean(errors)
    conf_int = bootstrap_ci(errors, conf_level)

    return errors, mean, conf_int


def mre(x_true, x_pred, conf_level=0.95):
    """
    Mean Relative Absolute Error (mre) with confidence interval calculation.
    
    Parameters:
    - x_true: array-like, true values
    - x_pred: array-like, predicted values
    - conf_level: float, confidence level for the interval
    
    Returns:
    - mean: float, the mre
    - conf_int: tuple, confidence interval for mre
    """
    errors = np.abs(x_true - x_pred) / np.abs(x_true)
    mean = np.mean(errors)
    conf_int = bootstrap_ci(errors, conf_level)

    return errors, mean, conf_int


def distances_errors(x_true, x_pred, conf_level=0.95):
    """
    Calculate the Mean Absolute Error (MAE) and Mean Relative Absolute Error (mre) between two sets of points.
    
    Parameters:
    - x_true: array-like, true values
    - x_pred: array-like, predicted values
    - conf_level: float, confidence level for the interval
    
    Returns:
    - mae_mean: float, the MAE
    - mae_conf_int: tuple, confidence interval for MAE
    - mre_mean: float, the mre
    - mre_conf_int: tuple, confidence interval for mre
    """
    dists_true = pdist(x_true.reshape((x_true.shape[0], -1)), metric='euclidean')
    dists_pred = pdist(x_pred.reshape((x_pred.shape[0], -1)), metric='euclidean')
    mae_vals, mae_mean, mae_conf_int = mae(dists_true, dists_pred, conf_level)
    mre_vals, mre_mean, mre_conf_int = mre(dists_true, dists_pred, conf_level)
    decile_distances, mae_by_decile = get_mean_value_by_decile(dists_true, mae_vals)
    _, mre_by_decile = get_mean_value_by_decile(dists_true, mre_vals)

    results = {
        'decile_distances': decile_distances,
        'mae_by_decile': mae_by_decile,
        'mae_mean': mae_mean,
        'mae_conf_int': mae_conf_int,
        'mre_by_decile': mre_by_decile,
        'mre_mean': mre_mean,
        'mre_conf_int': mre_conf_int
    }

    return results