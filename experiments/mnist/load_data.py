import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist


def get_datasets(npoints=2000, split=0.5, seed=123, noise=0.5, n_classes=6):
    np.random.seed(seed)
    # Load the images
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X = np.concatenate([X_train, X_test])
    y = np.concatenate([y_train, y_test])
    # Select n_classes first classes
    selection = y < n_classes
    X = X[selection]
    y = y[selection]
    # Shuffle the training data
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    # Select only the first npoints
    X = X[:npoints]
    y = y[:npoints]
    # Scale pixels to [0, 1] interval
    X = X / 255.0
    if noise > 0:
        # Add white noise
        X = X + np.random.normal(loc=0.0, scale=noise, size=X.shape)
        # Clip the pixel values in the [0, 1] interval
        X = np.clip(X, 0.0, 1.0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=seed)
    data_train = (X_train, y_train)
    data_test = (X_test, y_test)

    return data_train, data_test

