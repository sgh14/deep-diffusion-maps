import numpy as np
from typing import Dict
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist


def get_data(
        npoints: int=2000,
        split: float=0.5, 
        seed: int=123,
        noise: float=0,
        n_classes: int=6
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load and preprocess the MNIST dataset, then split it into training and test sets.

    This function loads the MNIST dataset, filters it to include only the specified number of classes,
    adds optional noise, and splits the data into training and test sets.

    Parameters:
        npoints (int): Total number of points to select from the dataset (default: 2000).
        split (float): Proportion of points to use for the test set (default: 0.5).
        seed (int): Random seed for reproducibility (default: 123).
        noise (float): Standard deviation of Gaussian noise added to the data (default: 0).
        n_classes (int): Number of classes to include from the dataset (default: 6).

    Returns:
        dict: Dictionary containing:
            - 'train': Dictionary with keys 'X' (image data) and 'y' (labels).
            - 'test': Dictionary with keys 'X' (image data) and 'y' (labels).
    """
    np.random.seed(seed)
    # Load the images
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X = np.concatenate([X_train, X_test])
    y = np.concatenate([y_train, y_test])
    # Select n_classes first classes
    selection = y < n_classes
    X, y = X[selection], y[selection]
    # Shuffle the training data
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]
    # Select only the first npoints
    X, y = X[:npoints], y[:npoints]
    # Scale pixels to [0, 1] interval
    X = X / 255.0
    if (noise > 0):
        # Add white noise
        X = X + np.random.normal(loc=0.0, scale=noise, size=X.shape)
        # Clip the pixel values in the [0, 1] interval
        X = np.clip(X, 0.0, 1.0)
    
    # Add a channel dimension to the images
    X = np.expand_dims(X, axis=-1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=seed)
    # Create the dictionary structure for the result
    data = {
        'train': {'X': X_train, 'y': y_train},
        'test': {'X': X_test, 'y': y_test}
    }

    return data