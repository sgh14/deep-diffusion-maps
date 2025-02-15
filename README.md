# deep-diffusion-maps


## Set up

```bash
conda env create -f experiments/environment.yml
```

## Usage

To run one the experiments, for example the one for Swiss Roll, execute

```bash
python -m experiments.swiss_roll.experiment -c experiments/swiss_roll/config.yml
python -m experiments.swiss_roll.plot_results -c experiments/swiss_roll/config.yml
```

To run all the experiments, execute

```bash
bash experiments/run_experiments.sh
```

You can modify the parameters of the experiments in their corresponding `config.yml` files.