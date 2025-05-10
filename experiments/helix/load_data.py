import numpy as np
from typing import Dict
from sklearn.model_selection import train_test_split

from experiments.utils.plots import normalize, colormap1D


def get_data(
        npoints: int=2000,
        split: float=0.5, 
        seed: int=123,
        noise: float=0
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Generate a 3D Helix dataset and split it into training and test sets.

    This function creates a 3D Helix dataset, assigns colors to each point based on its position
    in the manifold, and splits the data into training and test sets.

    Parameters:
        npoints (int): Total number of points to generate (default: 2000).
        split (float): Proportion of points to use for the test set (default: 0.5).
        seed (int): Random seed for reproducibility (default: 123).
        noise (float): Standard deviation of Gaussian noise added to the data (default: 0).

    Returns:
        dict: Dictionary containing:
            - 'train': Dictionary with keys 'X' (3D coordinates) and 'y' (RGB colors)
            - 'test': Dictionary with keys 'X' (3D coordinates) and 'y' (RGB colors)
    """
    np.random.seed(seed)
    theta = np.random.uniform(0, 2*np.pi, npoints)
    x1 = np.cos(theta)
    x2 = np.sin(2*theta)
    x3 = np.sin(3*theta)
    X = np.stack((x1, x2, x3), axis=1)
    if noise > 0:
        eps = np.random.normal(loc=0, scale=noise, size=(npoints, 3))
        X = X + eps
        
    y = theta
    # Generate colors for each point using the 1D colormap
    y = np.array([colormap1D(t) for t in normalize(theta)])
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=seed)
    # Create the dictionary structure for the result
    data = {
        'train': {'X': X_train, 'y': y_train},
        'test': {'X': X_test, 'y': y_test}
    }

    return data