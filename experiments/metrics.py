import numpy as np


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

    # Bootstrap for confidence interval
    bootstraps = []
    for _ in range(1000):
        sample_indices = np.random.choice(len(errors), len(errors), replace=True)
        sample_mean = np.mean(errors[sample_indices])
        bootstraps.append(sample_mean)

    lower_bound = np.percentile(bootstraps, (1 - conf_level) / 2 * 100)
    upper_bound = np.percentile(bootstraps, (1 + conf_level) / 2 * 100)
    conf_int = (lower_bound, upper_bound)

    return mean, conf_int
