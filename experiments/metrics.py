import numpy as np
from scipy.spatial.distance import pdist


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

    return mean, conf_int


def mrae(x_true, x_pred, conf_level=0.95):
    """
    Mean Relative Absolute Error (MRAE) with confidence interval calculation.
    
    Parameters:
    - x_true: array-like, true values
    - x_pred: array-like, predicted values
    - conf_level: float, confidence level for the interval
    
    Returns:
    - mean: float, the MRAE
    - conf_int: tuple, confidence interval for MRAE
    """
    errors = np.abs(x_true - x_pred) / np.abs(x_true)
    mean = np.mean(errors)
    conf_int = bootstrap_ci(errors, conf_level)

    return mean, conf_int


def distances_errors(x_true, x_pred, conf_level=0.95):
    """
    Calculate the Mean Absolute Error (MAE) and Mean Relative Absolute Error (MRAE) between two sets of points.
    
    Parameters:
    - x_true: array-like, true values
    - x_pred: array-like, predicted values
    - conf_level: float, confidence level for the interval
    
    Returns:
    - mae_mean: float, the MAE
    - mae_conf_int: tuple, confidence interval for MAE
    - mrae_mean: float, the MRAE
    - mrae_conf_int: tuple, confidence interval for MRAE
    """
    dists_true = pdist(x_true.reshape((x_true.shape[0], -1)), metric='euclidean')
    dists_pred = pdist(x_pred.reshape((x_pred.shape[0], -1)), metric='euclidean')
    mae_mean, mae_conf_int = mae(dists_true, dists_pred, conf_level)
    mrae_mean, mrae_conf_int = mrae(dists_true, dists_pred, conf_level)

    return (mae_mean, mae_conf_int), (mrae_mean, mrae_conf_int)