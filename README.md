# Machine learning prediction on microbial time series

This repository contains scripts for creating and analyzing machine learning models for the prediction of microbial time series data. The goal is
to predict changes in microbial communities and forecast trends.

## Testing phase

This is the testing ground for the updated model and workflow architecture. If you are testing this, please follow these instructions:

First you have to clone this repository to your machine. The code has been currently only tested on a linux system, so this would definitely be recommended at this point.
You can either clone the repo with:

    git clone "https://github.com/IKIM-Essen/Microbiome_TimeSeries.git"

Or you can directly download the repo as zip file form GitHub and then unpack it.

Eitherway, when you installed the code, please change into the main project directory "Microbiome_TimeSeries".

Then create a conda environment from the yaml file environment.yaml with:

    conda create --file environment.yaml -n YOUR_ENV_NAME

After that, activate it with:

    conda activate YOUR_ENV_NAME

Before you can test the whole thing, you need provide some data. There are currently three files that need to be provided. The main file is the timeseries.tsv. This holds the bacterial time series data from your analysis in the following format:

| taxonomy                  | timepoint1      | timepoint2      | timepoint3      |
|---------------------------|-----------------------------------------------------|
| genus name in Silva format| abundance1      | abundance2      | abundance3      |

Then there is the metadata.tsv which has the same format but holds all kinds of metadata or additional information to the data.

The third file is the taxa.tsv file. Here all the bacterial taxa on genus level should be listed. This is important, if you want to train and retrain the model on datasets potentially holding different bacteria, as the model needs to be trained on a fixed set of bacteria, even if they are only present in one dataset you want to analyze and not all of them.

To run the preprocessing of the data which creates a table that will be used as input please use the following command:

    python scripts/preprocessing.py --timeseries path/to/file --metadata /path/to/file --taxa path/to/file --include_metadata True --output path/to/output

If you do not set any of these flags, the code will use the default settings. The output should be a table located in results/tables/ .


## Features and Capabilities

The repository holds the code for creating, training and also retraining different kinds of models. You can decide on the model via the profile in the config directory. All models are capable of predicting one step into the future based on three previous time steps. LSTMs have been built to create a prediction
interval as base for outlier detection.
Calculated evaluation parameters include: MSE, RMSE, NRMSE
Model feature importance analysis has been done with SHAP.

## Datasets

The data used as input is 16S rRNA gene abundance data in form of a tsv file created from a BIOM file. The environmental model can process metadata as
well provided in a separate file.
Models have been trained and tested on published time series data from the following references: \
Caporaso et al., Genome Biology, 2011 \
David et al., Genome Biology, 2014 \
Kodera et al., Environ Microbiome, 2023 \
LaMartina et al., Microbiome, 2021 \
Dörr et al., Scientific Reports, 2025 \

## Installation

All necessary dependencies can be found in the environment.yaml and installed from there.

## Pipeline configuration with `config/profile.yaml`

The repository now supports a profile-driven workflow using `config/profile.yaml`.
This profile defines project metadata, input locations, output subdirectories, and model settings.

Example profile sections:

- `project`: contains `name`, `base_dir`, and `input_dir`.
- `paths`: defines where outputs are stored under the project base.
- `input_files`: lists the expected filenames inside `input_dir`.
- `parameters`: holds preprocessing and pipeline settings.
- `model_architecture` / `model_config`: can later be used to choose model variants.

All result paths are now built under:

```text
results/<project.name>/...
```

For example:

```text
results/tcn_residual_experiment_01/tables/
results/tcn_residual_experiment_01/models/
results/tcn_residual_experiment_01/intermediate/
```

## Running the full pipeline

A single entrypoint script has been added:

```bash
python scripts/run_pipeline.py
```

This will:

1. load `config/profile.yaml`
2. validate required profile sections
3. create output directories under `results/<project.name>/...`
4. run preprocessing, training, prediction, evaluation, and visualization stages

### Run only one stage

    ```bash
    python scripts/run_pipeline.py --stages preprocess
    ```

Supported stage names are:
- `preprocess`
- `train`
- `predict`
- `evaluate`
- `visualize`
- `retraining`

### Dry run

To inspect the commands without executing them:

    ```bash
    python scripts/run_pipeline.py --dry-run
    ```

## Script reference

### `scripts/preprocessing.py`

Creates the complete input dataframe from `timeseries.tsv`, `metadata.tsv`, and `taxa.tsv`.
By default it writes to `results/<project.name>/tables/complete_df.csv`, and it can also save split files to `results/<project.name>/intermediate/`.

### `scripts/training.py`

Loads training/validation splits and fits the current model architecture.
The current implementation trains a TCN model and an LSTM residual model, and saves them to the project model directory.

### `scripts/prediction.py`

Loads the trained models and split arrays, then writes prediction arrays to:

    ```text
    results/<project.name>/intermediate/predictions.npz
    ```

### `scripts/evaluation.py`

Loads prediction outputs and splits to compute evaluation metrics, writing them to:

    ```text
    results/<project.name>/tables/evaluation_metrics.tsv
    ```

### `scripts/plot_taxa_violations.py`

Creates an interactive Plotly HTML file showing the selected taxa time series and any violation table.

## Notes

- The new profile-driven pipeline reduces hardcoded paths and centralizes project setup.
- `src/utils/config.py` now includes helpers to load and validate profile YAML files.
- The code is still primarily tested on Linux.

## Cite

Please feel free to use the software. If you do so and it results in a scientific publication, please cite the papers to this repository:

Dörr, AK., Imangaliyev, S., Karadeniz, U. et al. Distinguishing critical microbial community shifts from normal temporal variability in human and environmental ecosystems. Sci Rep 15, 16934 (2025). https://doi.org/10.1038/s41598-025-01781-x

Dörr, AK., Schmidt, T., Schoth, J., Kraiselburd, I., Meyer, F. (2026). Which Metadata Matters? Evaluating Predictive Features for Environmental Time Series. In: Bruno, P., Calimeri, F., Cauteruccio, F., Dragoni, M., Stella, F., Terracina, G. (eds) Artificial Intelligence for Healthcare, and Hybrid Models for Coupling Deductive and Inductive Reasoning. HC_AIxIA_HYDRA 2025. Communications in Computer and Information Science, vol 2830. Springer, Cham. https://doi.org/10.1007/978-3-032-16708-8_20
