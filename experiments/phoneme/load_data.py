import numpy as np
from typing import Dict
from sklearn.model_selection import train_test_split
from skfda import datasets


def get_data(
        split: float=0.5, 
        seed: int=123,
        noise: float=0
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load the Phoneme dataset, optionally add noise, and split it into training and test sets.

    This function fetches the Phoneme dataset using scikit-fda, optionally adds Gaussian noise to the data,
    and splits it into training and test sets.

    Parameters:
        split (float): Proportion of points to use for the test set (default: 0.5).
        seed (int): Random seed for reproducibility (default: 123).
        noise (float): Standard deviation of Gaussian noise added to the data (default: 0).

    Returns:
        dict: Dictionary containing:
            - 'train': Dictionary with keys 'X' (data matrix) and 'y' (labels)
            - 'test': Dictionary with keys 'X' (data matrix) and 'y' (labels)
    """
    n_points = 150
    np.random.seed(seed)
    X, y = datasets.fetch_phoneme(return_X_y=True)
    X = X.data_matrix[:, :n_points]
    if noise > 0:
        X = X + np.random.normal(loc=0.0, scale=noise, size=X.shape)

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split, stratify=y, random_state=seed, shuffle=True
    )
    # Create the dictionary structure for the result
    data = {
        'train': {'X': X_train, 'y': y_train},
        'test': {'X': X_test, 'y': y_test}
    }

    return data