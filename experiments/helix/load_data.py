import numpy as np
from sklearn.model_selection import train_test_split


def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def my_colormap1D(x, c1=(0.75, 0, 0.75), c2=(0, 0.75, 0.75)):
    # Calculate the RGB values based on interpolation
    color = np.array(c1) * (1 - x) + np.array(c2) * x

    return color


def my_colormap2D(x, y):
    # Define colors in RGB
    bottom_left = (0.5, 0, 0.5) # dark magenta
    bottom_right = (0, 0.5, 0.5) # dark cyan
    top_left = (1, 0, 1) # magenta
    top_right = (0, 1, 1) # cyan

    # Calculate the RGB values based on interpolation
    top_color = np.array(top_left) * (1 - x) + np.array(top_right) * x
    bottom_color = np.array(bottom_left) * (1 - x) + np.array(bottom_right) * x

    return top_color * (1 - y) + bottom_color * y


def get_datasets(npoints=2000, split=0.5, seed=123, noise=0):
    np.random.seed(seed)
    # theta = np.linspace(0, 2*np.pi, npoints)
    theta = np.random.uniform(0, 2*np.pi, npoints)
    x1 = np.cos(theta)
    x2 = np.sin(2*theta)
    x3 = np.sin(3*theta)
    X = np.stack((x1, x2, x3), axis=1)
    if noise > 0:
        eps = np.random.normal(loc=0, scale=noise, size=(npoints, 3))
        X = X + eps
        
    y = theta
    y = np.array([my_colormap1D(t) for t in normalize(theta)])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=seed)
    data_train = (X_train, y_train)
    data_test = (X_test, y_test)

    return data_train, data_test