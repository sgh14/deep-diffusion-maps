import numpy as np
from typing import Dict
from sklearn.datasets import make_s_curve
from sklearn.model_selection import train_test_split

from experiments.utils.plots import normalize, colormap2D


def get_data(
        npoints: int=2000,
        split: float=0.5, 
        seed: int=123,
        noise: float=0
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Generate a colored S Curve dataset and split it into training and test sets.
    
    This function creates a 3D S Curve dataset using scikit-learn, assigns colors to
    each point based on its position in the manifold, and splits the data into training
    and test sets.
    
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
    # Generate the S Curve dataset
    X, color = make_s_curve(npoints, random_state=seed, noise=noise)
    
    # Normalize the principal curve parameter (color) and the height
    dimension_1 = normalize(color)  # Normalized position along the S (0 to 1)
    dimension_2 = normalize(X[:, 1])  # Normalized height of the Swiss Roll (0 to 1)
    
    # Generate colors for each point using the 2D colormap
    y = np.array([colormap2D(x, y) for x, y in zip(dimension_1, dimension_2)])
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=seed)
    # Create the dictionary structure for the result
    data = {
        'train': {'X': X_train, 'y': y_train},
        'test': {'X': X_test, 'y': y_test}
    }

    return data