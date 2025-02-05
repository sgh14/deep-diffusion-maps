import numpy as np
from sklearn.model_selection import train_test_split
from skfda import datasets


def get_datasets(split=0.5, seed=123, noise=0.5):
    n_points = 150
    np.random.seed(seed)
    X, y = datasets.fetch_phoneme(return_X_y=True)
    # new_points = X.grid_points[0][:n_points]
    # new_data = X.data_matrix[:, :n_points]
    # X = X.copy(
    #     grid_points=new_points,
    #     data_matrix=new_data,
    #     domain_range=(np.min(new_points), np.max(new_points)),
    # )
    # X = X(np.linspace(*domain_range, 128))
    X = X.data_matrix[:, :n_points]
    if noise > 0:
        X = X + np.random.normal(loc=0.0, scale=noise, size=X.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split, stratify=y, random_state=seed, shuffle=True
    )
    data_train = (X_train, y_train)
    data_test = (X_test, y_test)

    return data_train, data_test
