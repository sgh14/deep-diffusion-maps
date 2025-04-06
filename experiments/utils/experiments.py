import os
import h5py
import numpy as np
from scipy.spatial.distance import pdist
from typing import Tuple, Dict, Any
import argparse
import yaml
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import History

from src.diffusionloss import DiffusionLoss
from diffusionmaps import DiffusionMaps
from experiments.utils.plots import colormap1D, normalize
from experiments.utils.metrics import distances_errors


def load_config() -> Tuple[Dict[str, Any], str]:
    """
    Load the configuration from a YAML file specified via command-line arguments.

    This function parses command-line arguments to locate a YAML configuration file,
    reads the file, and returns its contents as a dictionary along with the file path.

    Returns:
        Tuple[Dict[str, Any], str]:
            - config (Dict[str, Any]): Configuration parameters loaded from the YAML file.
            - config_path (str): Path to the YAML configuration file.
    """
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to the YAML configuration file."
    )
    args = parser.parse_args()
    config_path = args.config

    # Load Configuration
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config, config_path


def get_sigma(X: np.ndarray, q: float = 0.5) -> float:
    """
    Estimate a scale parameter (sigma) from the pairwise distances between samples.

    Parameters:
        X (np.ndarray): Input array of samples.
        q (float): Quantile to use for sigma estimation (default is 0.5, i.e., median).

    Returns:
        float: Estimated sigma value.
    """
    X_flat = X.reshape((X.shape[0], -1))
    distances = pdist(X_flat, metric='euclidean')
    sigma = np.quantile(distances, q)
    
    return sigma


def log_gaussian_density(x: np.ndarray, mean: float, var: float) -> np.ndarray:
    """
    Compute the log-probability density under a Gaussian distribution.

    Parameters:
        x (np.ndarray): Input values.
        mean (float): Mean of the Gaussian.
        var (float): Variance of the Gaussian.

    Returns:
        np.ndarray: Log-probability of each value in x.
    """
    return -0.5 * np.log(2 * np.pi * var) - (x - mean) ** 2 / (2 * var)


def log_likelihood(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the total log-likelihood assuming x and y are from the same Gaussian distribution.

    Parameters:
        x (np.ndarray): First set of samples.
        y (np.ndarray): Second set of samples.

    Returns:
        float: Total log-likelihood.
    """
    p, q = len(x), len(y)
    mean_x = np.mean(x)
    var_x = np.var(x, ddof=1 if p > 1 else 0)
    mean_y = np.mean(y)
    var_y = np.var(y, ddof=1 if q > 1 else 0)

    pooled_var = ((p - 1) * var_x + (q - 1) * var_y) / (p + q - 2)
    log_likelihood_x = np.sum(log_gaussian_density(x, mean_x, pooled_var))
    log_likelihood_y = np.sum(log_gaussian_density(y, mean_y, pooled_var))

    return log_likelihood_x + log_likelihood_y


def log_likelihood_curve(eigenvalues: np.ndarray, max_components: int = 25) -> np.ndarray:
    """
    Compute log-likelihoods for varying numbers of eigenvalue components.

    Parameters:
        eigenvalues (np.ndarray): Array of eigenvalues.
        max_components (int): Maximum number of components to include.

    Returns:
        np.ndarray: Log-likelihood values for 1 to `max_components` components.
    """
    n_components_vals = np.arange(1, max_components + 1)
    l_vals = np.empty(len(n_components_vals))
    for i, n_components in enumerate(n_components_vals):
        x = eigenvalues[:n_components]
        y = eigenvalues[n_components:]
        l_vals[i] = log_likelihood(x, y)

    return l_vals


def diffusion_maps_experiment(
    data: Dict[str, Dict[str, np.ndarray]], 
    sigma: float, 
    n_components: int, 
    steps: int, 
    alpha: float
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Run the standard diffusion maps algorithm on the given data.

    Parameters:
        data (Dict[str, Dict[str, np.ndarray]]): Dictionary containing 'train' and 'test' data, each with 'X' key.
        sigma (float): Kernel bandwidth parameter.
        n_components (int): Number of dimensions to reduce to.
        steps (int): Number of diffusion steps (powers of the transition matrix).
        alpha (float): Alpha parameter for diffusion maps.

    Returns:
        Dict[str, Dict[str, np.ndarray]]: Dictionary containing reduced representations for train and test data.
    """
    n_train = len(data['train']['X'])
    # Combine train and test data for diffusion maps
    X = np.vstack([data['train']['X'], data['test']['X']])
    
    # Initialize and fit diffusion maps
    dm = DiffusionMaps(sigma=sigma, n_components=n_components, steps=steps, alpha=alpha)
    X_red = dm.fit_transform(X)
    
    # Split the results back into train and test
    results = {
        'train': {'X_red': X_red[:n_train]},
        'test': {'X_red': X_red[n_train:]}
    }
    
    return results


def deep_diffusion_maps_experiment(
    data: Dict[str, Dict[str, np.ndarray]], 
    sigma: float, 
    n_components: int, 
    steps: int, 
    alpha: float, 
    encoder_builder: Any,
    encoder_config: Dict[str, Any], 
    reference_results: Dict[str, Dict[str, np.ndarray]]
) -> Tuple[Dict[str, Dict[str, Any]], Model, History]:
    """
    Run the deep diffusion maps algorithm using a neural network encoder.

    Parameters:
        data (Dict[str, Dict[str, np.ndarray]]): Dictionary containing 'train' and 'test' data, each with 'X' key.
        sigma (float): Kernel bandwidth parameter.
        n_components (int): Number of dimensions to reduce to.
        steps (int): Number of diffusion steps.
        alpha (float): Alpha parameter for diffusion maps.
        encoder_builder (Any): Function to build the neural network encoder.
        encoder_config (Dict[str, Any]): Configuration for the neural network encoder.
        reference_results (Dict[str, Dict[str, np.ndarray]]): Results from standard diffusion maps for comparison.

    Returns:
        Tuple[Dict[str, Dict[str, Any]], Model, History]:
            - Results dict containing reduced representations and metrics.
            - Trained encoder model.
            - Training history.
    """
    # Build neural network encoder
    encoder = encoder_builder(
        input_shape=data['train']['X'].shape[1:],
        n_components=n_components,
        **encoder_config['architecture']
    )
    encoder.summary()
    
    # Create diffusion loss function
    loss = DiffusionLoss(data['train']['X'], sigma=sigma, steps=steps, alpha=alpha)
    optimizer = Adam(**encoder_config['optimizer'])
    encoder.compile(loss=loss, optimizer=optimizer)

    # Train the encoder
    indices = np.arange(data['train']['X'].shape[0])
    history = encoder.fit(
        x=data['train']['X'],
        y=indices,
        **encoder_config['training']
    )
    
    # Generate embeddings for train and test data
    X_train_red = encoder(data['train']['X']).numpy()
    X_test_red = encoder(data['test']['X']).numpy()
    
    # Calculate metrics against reference results
    metrics_train = distances_errors(reference_results['train']['X_red'], X_train_red)
    metrics_test = distances_errors(reference_results['test']['X_red'], X_test_red)
    
    # Compile results
    results = {
        'train': {'X_red': X_train_red, **metrics_train},
        'test': {'X_red': X_test_red, **metrics_test}
    }

    return results, encoder, history


def nystrom_experiment(
    data: Dict[str, Dict[str, np.ndarray]], 
    sigma: float, 
    n_components: int, 
    steps: int, 
    alpha: float, 
    reference_results: Dict[str, Dict[str, np.ndarray]],
    reference_coords: np.ndarray
) -> Dict[str, Dict[str, Any]]:
    """
    Run the Nyström extension experiment for diffusion maps.

    Parameters:
        data (Dict[str, Dict[str, np.ndarray]]): Dictionary containing 'train' and 'test' data, each with 'X' key.
        sigma (float): Kernel bandwidth parameter.
        n_components (int): Number of dimensions to reduce to.
        steps (int): Number of diffusion steps.
        alpha (float): Alpha parameter for diffusion maps.
        reference_results (Dict[str, Dict[str, np.ndarray]]): Results from standard diffusion maps for comparison.
        reference_coords (np.ndarray): Reference coordinates for visualization.

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary containing reduced representations and visualization data.
    """
    # Initialize diffusion maps
    dm = DiffusionMaps(sigma=sigma, n_components=n_components, steps=steps, alpha=alpha)
    
    # Fit on training data and transform test data using Nyström extension
    X_train_red = dm.fit_transform(data['train']['X'])
    X_test_red = dm.transform(data['test']['X'])
    
    # Calculate metrics against reference results
    metrics_train = distances_errors(reference_results['train']['X_red'], X_train_red)
    metrics_test = distances_errors(reference_results['test']['X_red'], X_test_red)

    # Extract eigenvalues for log-likelihood curve
    eigenvalues = dm.lambdas[1:]**steps
    log_likelihood = log_likelihood_curve(eigenvalues)

    # Calculate diffusion probabilities and distances for visualization
    P_train = dm.transition_probabilities(dm.W)
    if steps > 1:
        P_train = np.linalg.matrix_power(P_train, steps)

    D_train = dm.diffusion_distances(P_train, dm.pi)
    point = np.argmin(np.linalg.norm(data['train']['X'] - reference_coords, axis=1))
    # Create color mappings for visualization, focusing on a specific point
    colors = [(0, 0, 0), (0, 0.75, 0.75), (0.75, 0, 0.75), (0.72, 0.53, 0.05)]
    P_color_train = np.array([colormap1D(x, colors[0], colors[1]) for x in normalize(P_train[:, point])])
    D_color_train = np.array([colormap1D(x, colors[0], colors[2]) for x in normalize(D_train[:, point])])
    P_color_train[point] = colors[3]
    D_color_train[point] = colors[3]

    # Compile results
    results = {
        'train': {
            'X_red': X_train_red, 
            'eigenvalues': eigenvalues,
            'log_likelihood': log_likelihood,
            'P_color': P_color_train,
            'D_color': D_color_train,
            **metrics_train
        },
        'test': {'X_red': X_test_red, **metrics_test}
    }

    return results


def save_encoder(encoder: Model, history: History, output_dir: str) -> None:
    """
    Save the trained encoder model and its training history.

    Parameters:
        encoder (Model): Trained encoder model.
        history (History): Training history of the encoder.
        output_dir (str): Directory to save the model and history.

    Returns:
        None
    """
    # Save trained encoder model
    encoder.save(os.path.join(output_dir, 'encoder.keras'))
    
    # Save training history
    with h5py.File(os.path.join(output_dir, 'history.h5'), 'w') as file:
        for key, value in history.history.items():
            file.create_dataset(key, data=value)


def save_results(results: Dict[str, Dict[str, Any]], hyperparameters: Dict[str, Any], output_dir: str) -> None:
    """
    Save all experiment results and hyperparameters.

    Parameters:
        results (Dict[str, Dict[str, Any]]): Experiment results to save.
        hyperparameters (Dict[str, Any]): Hyperparameters used in the experiment.
        output_dir (str): Directory to save the results.

    Returns:
        None
    """
    # Save all experiment results
    with h5py.File(os.path.join(output_dir, 'results.h5'), "w") as file:
        # Save hyperparameters without compression (small scalars)
        file.create_group("hyperparameters")
        for key, value in hyperparameters.items():
            file['hyperparameters'].create_dataset(key, data=value)

        # Create the groups and datasets with compression for large arrays
        for group_name, group_dict in results.items():
            file.create_group(group_name)
            for subset_name, subset_dict in group_dict.items():
                file[group_name].create_group(subset_name)
                for key, value in subset_dict.items():
                    if hasattr(value, 'shape') and len(value) > 1:
                        file[group_name][subset_name].create_dataset(key, data=value, compression='gzip')
                    else:
                        file[group_name][subset_name].create_dataset(key, data=value)