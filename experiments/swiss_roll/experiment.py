import os
import numpy as np
from experiments.utils.experiments import load_config, get_sigma, diffusion_maps_experiment, deep_diffusion_maps_experiment, nystrom_experiment, save_encoder, save_results
from experiments.swiss_roll.load_data import get_data
from experiments.utils.models import build_encoder


def main() -> None:
    """
    Main function to run all experiments.
    
    The function:
    1. Loads configuration from a YAML file
    2. Creates output directories
    3. Loads and prepares data
    4. Runs the three types of experiments
    5. Saves results and trained models
    """
    # Load configuration
    config, config_path = load_config()
    hyperparameters = config['diffusion_maps']

    # Create directories for output
    root = config['output_dir']
    experiment = '_'.join([k + '_' + str(v) for k, v in hyperparameters.items()])
    output_dir = os.path.join(root, experiment)
    os.makedirs(output_dir, exist_ok=True)
    # Copy configuration file to output directory for reference
    os.system(f'cp {config_path} {output_dir}')

    # Load the data
    data = get_data(
        npoints=config['data']['npoints'],
        split=config['data']['split'],
        seed=config['data']['seed'],
        noise=config['data']['noise']
    )

    # Calculate sigma parameter based on quantile if not explicitly provided
    hyperparameters['sigma'] = get_sigma(data['train']['X'], hyperparameters['quantile'])
    
    # Run standard diffusion maps experiment
    dm_results = diffusion_maps_experiment(
        data=data,
        sigma=hyperparameters['sigma'],
        n_components=hyperparameters['n_components'],
        steps=hyperparameters['steps'],
        alpha=hyperparameters['alpha']
    )
    
    # Run deep diffusion maps experiment
    ddm_results, encoder, history = deep_diffusion_maps_experiment(
        data=data,
        sigma=hyperparameters['sigma'],
        n_components=hyperparameters['n_components'],
        steps=hyperparameters['steps'],
        alpha=hyperparameters['alpha'],
        encoder_builder=build_encoder,
        encoder_config=config['encoder'],
        reference_results=dm_results
    )
    
    # Run Nyström extension experiment
    nystrom_results = nystrom_experiment(
        data=data,
        sigma=hyperparameters['sigma'],
        n_components=hyperparameters['n_components'],
        steps=hyperparameters['steps'],
        alpha=hyperparameters['alpha'],
        reference_results=dm_results,
        reference_coords=np.array([0, -1, -15])
    )
    
    # Aggregate all results
    results = {
        'original': data,
        'diffusion_maps': dm_results,
        'deep_diffusion_maps': ddm_results,
        'nystrom': nystrom_results
    }

    save_encoder(encoder, history, output_dir)
    save_results(results, hyperparameters, output_dir)
    

if __name__ == "__main__":
    main()