# Machine learning for microbial time series

This repository provides a workflow for training and evaluating machine learning models on microbial abundance time series. Its aim is to predict changes in microbial communities and identify observations that fall outside model-derived prediction intervals.

The workflow starts with a microbial abundance table, a reference list of taxa, and a matching metadata table. Metadata can describe environmental conditions or other sample characteristics. Outputs include trained models, predictions, evaluation metrics, prediction intervals, and an interactive plot for exploring potential anomalies.

The workflow is under active development. Read [Project status](#project-status) and [Current execution limitations](#current-execution-limitations) before running an analysis. The input examples illustrate file structure; they are not a validated end-to-end training example.

## Workflow overview

| Step | What happens | Main result |
| --- | --- | --- |
| Prepare inputs | Supply abundance data, matching metadata, and a reference taxa list. | Three input files |
| Configure the experiment | Set the experiment name, file locations, and model architecture in `config/profile.yaml`. | An experiment profile |
| Preprocess | Align taxa to the reference list, assemble the input table, scale values, and split the data into training, validation, and test sets. | Prepared table, scaler, and data splits |
| Train | Fit the selected model using the training and validation sets. | Saved models |
| Predict | Apply the trained models to the prepared data splits. | Prediction arrays |
| Evaluate | Compare predictions with observed values. | Error metrics |
| Calculate intervals | Use an ensemble of models to estimate prediction bounds and identify observations outside them. | Prediction intervals and an anomaly table |
| Visualize | Explore observed values, predictions, intervals, and anomalies by taxon. | An interactive HTML plot |

Training fits model parameters. Validation provides a separate subset during model development. The test set is the later part of the time series used to assess predictions. The standard preprocessing defaults allocate 70% of observations to training, 10% to validation, and the remaining 20% to testing, in chronological order.

Throughout this README, a **taxon** is one microbial group represented by a taxonomy label; **taxa** is the plural. A **timepoint** is one sampling observation. An **anomaly** is an observation outside a model-derived prediction interval, also called an interval violation in parts of the code.

## Installation

Clone the repository and change into its main directory:

```bash
git clone https://github.com/IKIM-Essen/Microbiome_TimeSeries.git
cd Microbiome_TimeSeries
```

Alternatively, download the repository as a ZIP file from GitHub, extract it, and open the extracted directory in your terminal.

Create and activate a Conda environment using the supplied dependency file:

```bash
conda env create --file environment.yaml --name microbiome-timeseries
conda activate microbiome-timeseries
```

The supplied environment contains Linux-specific package builds. Installation on macOS or Windows has not been established. Run the commands in this README from the repository's main directory so that relative paths such as `config/profile.yaml` resolve correctly.

## Prepare your input data

By default, the workflow looks for these files:

```text
data/
├── timeseries.tsv
├── metadata.tsv
└── taxa.tsv
```

You can use different locations and filenames by editing `config/profile.yaml`.

The examples below use the same two taxa and three sampling dates throughout. They illustrate file structure only: three observations are insufficient for the default training, validation, and test split. They are not a runnable training dataset.

### Microbial abundances: `timeseries.tsv`

Use a tab-separated table with taxa in rows and sampling timepoints in columns. The first row contains column headers; the first column contains taxonomy labels through the genus level in SILVA-style format.

<!-- markdownlint-disable MD010 -->

```tsv
taxonomy	2025-01-01	2025-01-02	2025-01-03
d__Bacteria; p__Bacteroidota; c__Bacteroidia; o__Bacteroidales; f__Bacteroidaceae; g__Bacteroides	0.60	0.55	0.70
d__Bacteria; p__Bacillota; c__Clostridia; o__Lachnospirales; f__Lachnospiraceae; g__Blautia	0.40	0.45	0.30
```

<!-- markdownlint-enable MD010 -->

These example values are relative abundances expressed as fractions. Use one consistent abundance scale throughout your analysis and document it when interpreting errors and intervals. Preprocessing applies numerical scaling for model fitting; it does not convert raw counts into relative abundances.

Preprocessing sums duplicate taxonomy labels, removes labels containing `Unassigned`, strips species suffixes matching `; s__`, and aligns the remaining taxa to the reference list. Taxa outside that list are excluded; reference taxa absent from the abundance table are filled with zero.

### Sample information: `metadata.tsv`

Use a tab-separated table with metadata variables in rows and sampling timepoints in columns. The headers must match the abundance table, in the same order.

<!-- markdownlint-disable MD010 -->

```tsv
variable	2025-01-01	2025-01-02	2025-01-03
temperature	18.2	18.5	19.1
pH	7.1	7.0	7.2
```

<!-- markdownlint-enable MD010 -->

For example, `temperature` and `pH` describe the conditions at each sampling timepoint. Use numeric values for ordinary model inputs. The repository also contains intended month and season encodings and a list of fields to drop in `config/encodings.yaml`; arbitrary text categories are not automatically encoded.

The metadata file is currently read even when metadata features are excluded from the model. Keep a matching metadata table available. The inclusion setting also has a known implementation issue described below.

### Reference taxa: `taxa.tsv`

Provide one taxonomy label per line, with no header or abundance columns:

```text
d__Bacteria; p__Bacteroidota; c__Bacteroidia; o__Bacteroidales; f__Bacteroidaceae; g__Bacteroides
d__Bacteria; p__Bacillota; c__Clostridia; o__Lachnospirales; f__Lachnospiraceae; g__Blautia
```

The labels must match the genus-level labels used in the abundance table after preprocessing. This file defines the model's taxa and their order. Keep the same reference list when applying or retraining an existing model on another dataset, including taxa absent from that dataset.

### Check the inputs before analysis

- Use actual tab characters to separate abundance and metadata columns.
- Use identical, unique timepoint headers in the abundance and metadata tables, in the same order.
- Use sortable time labels, such as ISO dates (`YYYY-MM-DD`). The prepared table is sorted by its `Time` column; dates are also used by the visualization.
- Inspect missing values before analysis. Preprocessing replaces missing abundance values and metadata values with zero, which may not be an appropriate scientific interpretation of missing measurements.
- Record abundance units and sampling frequency. There is no explicit resampling step for irregularly spaced observations.
- Supply enough observations for nonempty training, validation, and test sets. A recommended minimum dataset size has not yet been established.

## Configure your experiment

The pipeline runner reads `config/profile.yaml`. Start by checking the experiment name, input locations, metadata setting, and model architecture.

The following excerpt explains those settings; retain the other sections of the supplied profile:

```yaml
project:
  name: my_experiment       # Name of the experiment's output directory.
  base_dir: results         # Parent directory for experiment outputs.
  input_dir: data           # Directory containing the three input files.

input_files:
  timeseries: timeseries.tsv
  metadata: metadata.tsv
  taxa: taxa.tsv

parameters:
  include_metadata: True    # Intended to include metadata as model features; see limitations.
  number_ensemble_model: 50 # Number of models used to estimate prediction intervals.

model_architecture: tcn_lstm
```

`model_architecture` selects the model to train. The code contains options for `tcn_lstm` (a temporal convolutional network with an LSTM residual model), `lstm`, `attention`, and `metadata_parallel`. These options use different preprocessing and model paths; their end-to-end compatibility still needs validation.

The `paths` section names output subdirectories such as `models`, `tables`, `intermediate`, and `figures`. Most readers can keep these names unchanged. With the example above, the runner places experiment outputs under `results/my_experiment/`. The supplied profile instead uses the experiment name `tcn_residual_experiment_01`.

The profile also contains training and validation proportions and a retraining setting. These are not currently connected to the runner's stage commands; see the limitations below before changing them.

## Run pipeline stages

Use `scripts/run_pipeline.py` to build stage commands from the profile. Stage selection is currently required. For example, to preprocess the input files:

```bash
python scripts/run_pipeline.py --stages preprocess
```

To select several stages, separate their names with commas:

```bash
python scripts/run_pipeline.py --stages preprocess,train,predict,evaluate
```

The implemented stages are `preprocess`, `train`, `predict`, `evaluate`, `interval`, and `visualize`. They execute in that fixed order. Later stages depend on files created by earlier stages, so selecting `predict` alone requires existing models and data splits. Visualization also needs the interval and anomaly outputs.

To inspect the generated commands without executing the stages:

```bash
python scripts/run_pipeline.py --stages preprocess,train,predict,evaluate,interval,visualize --dry-run
```

A dry run still loads the profile and creates the configured output directories. It does not verify input contents, model compatibility, or successful execution. Use `--profile path/to/profile.yaml` to supply another experiment profile.

### Current execution limitations

The following issues need attention before treating this as a verified end-to-end workflow:

- **Metadata inclusion:** the preprocessing CLI receives the inclusion setting as text, while the combining function checks for Boolean `True`. Setting `include_metadata: True` therefore does not enable the metadata columns in the combined table through this route. The metadata file is still read when inclusion is disabled.
- **Split settings:** the runner does not pass the profile's `training_percentage` and `validation_percentage` to preprocessing. The script's defaults currently apply instead.
- **Retraining:** the runner accepts the name `retrain` but does not dispatch a retraining stage. The profile's `retraining` setting does not trigger retraining. `scripts/retraining.py` is a separate entrypoint.
- **Visualization paths:** the runner does not pass the experiment-specific complete-table and taxa-mapping paths to the plotter, which falls back to standalone defaults. These paths need to be supplied explicitly when invoking the plotter directly, or corrected in the runner.
- **Forecasting setup:** the intended task is prediction over time. The standard splitting path reshapes features into three positions without explicitly constructing lags from preceding observations; the attention preprocessing path constructs lagged features separately. Input lags and future targets need verification across architectures before interpreting outputs as next-timepoint forecasts.

## Understand the results

For the supplied profile, the runner uses `results/tcn_residual_experiment_01/` as the experiment directory. The table below gives paths relative to that directory and describes the outputs of successful stages.

| Output | What it contains | How to use it |
| --- | --- | --- |
| `tables/complete_df.csv` | Prepared observations with a `Time` column and numbered taxa columns such as `Target1`. | Check that dates and abundance values were assembled as expected. |
| `intermediate/dic_TargTax.pkl` | A mapping from numbered target columns to taxonomy labels. | Connect model targets back to taxa. |
| `intermediate/splits.npz` | Arrays used for training, validation, and testing. | Inputs to model training and prediction. |
| `models/` | Saved model files and the fitted scaler. | Reuse the models and the same numerical scaling. |
| `intermediate/predictions.npz` | Predictions for the training, validation, and test sets. | Inputs to evaluation and plotting. |
| `tables/evaluation_metrics.tsv` | Training and test MAE, RMSE, and NRMSE. | Assess prediction errors; lower values indicate smaller errors. |
| `tables/prediction_interval.tsv` | Per-taxon interval bounds and mean predictions, indexed by timepoint. | Examine the range estimated by the ensemble. |
| `tables/prediction_interval_anomalies.tsv` | Observations outside the interval, including the taxon, time, observed value, bounds, and direction of the violation. | Identify observations for further investigation. |
| `figures/plot_taxa_anomalies.html` | An interactive Plotly view with a taxon selector. | Open in a web browser to explore the results visually. |

Start with `complete_df.csv` to check the prepared data, then inspect the evaluation table and interactive plot.

Mean absolute error (MAE) measures average absolute prediction error. Root mean squared error (RMSE) gives greater weight to larger errors. Normalized RMSE (NRMSE) divides RMSE by the standard deviation of the observed values in the evaluated set. MAE and RMSE are computed after inverse scaling, so their units depend on the abundance scale supplied as input.

The ensemble produces lower and upper prediction bounds for each taxon. An anomaly is an observation above the upper bound or below the lower bound in the test period. This flags a departure from the model-derived range; it does not by itself establish a biological community shift or its cause. Interpretation depends on input quality, model performance, and the interval method's assumptions and coverage.

## Script reference

The runner supplies experiment-specific paths to the stage scripts. Running a script directly does not automatically apply all profile settings; standalone defaults commonly point to `results/tables/`, `results/models/`, or `results/intermediate/` without an experiment name.

| Script | Role |
| --- | --- |
| `scripts/run_pipeline.py` | Read the experiment profile and invoke the selected stages. |
| `scripts/preprocessing.py` | Create the prepared table, taxa mapping, scaler, and data splits. |
| `scripts/training.py` | Fit the selected model architecture. |
| `scripts/prediction.py` | Load trained models and generate prediction arrays. |
| `scripts/evaluation.py` | Compare predictions with observations and save error metrics. |
| `scripts/interval.py` | Estimate prediction intervals and detect observations outside them. |
| `scripts/plot_taxa_anomalies.py` | Create the interactive HTML plot. |
| `scripts/retraining.py` | Provide a separate workflow for adapting existing models to new data. |

## Datasets and historical analyses

The input abundance data are derived from 16S rRNA gene analyses, for example by converting a BIOM table into a tab-separated table with the structure described above.

Previous analyses associated with this repository used published time series from:

- Caporaso et al., *Genome Biology*, 2011
- David et al., *Genome Biology*, 2014
- Kodera et al., *Environmental Microbiome*, 2023
- LaMartina et al., *Microbiome*, 2021
- Dörr et al., *Scientific Reports*, 2025

Historical scripts are retained in [`old/`](old/), including alternative model architectures and SHAP feature importance analyses. The current workflow uses `scripts/`, `src/`, and `config/`; its stage entrypoints are listed in the [script reference](#script-reference). SHAP analysis is not a stage in the current pipeline runner.

The publications below describe earlier analyses. Their reported results do not establish that the current pipeline reproduces those analyses or that every current model architecture has been validated.

## Project status

The profile-driven workflow is being developed and has primarily been tested on Linux. The supplied environment contains Linux-specific builds; macOS and Windows support has not been established.

This README describes the stages and outputs present in the current code. End-to-end execution, compatibility across model architectures, and the forecasting setup still need validation. The [execution limitations](#current-execution-limitations) identify known discrepancies between the configuration and software behavior.

## License

This software is released under the [MIT License](LICENSE).

## Citation

If you use this software in a scientific publication, please cite the following papers:

Dörr, AK., Imangaliyev, S., Karadeniz, U. et al. Distinguishing critical microbial community shifts from normal temporal variability in human and environmental ecosystems. Sci Rep 15, 16934 (2025). doi.org/10.1038/s41598-025-01781-x

Dörr, AK., Schmidt, T., Schoth, J., Kraiselburd, I., Meyer, F. (2026). Which Metadata Matters? Evaluating Predictive Features for Environmental Time Series.
In: Bruno, P., Calimeri, F., Cauteruccio, F., Dragoni, M., Stella, F., Terracina, G. (eds) Artificial Intelligence for Healthcare, and Hybrid Models for Coupling Deductive and Inductive Reasoning. HC_AIxIA_HYDRA 2025. Communications in Computer and Information Science, vol 2830. Springer, Cham. doi.org/10.1007/978-3-032-16708-8_20
