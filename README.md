# Machine learning for microbial time series

This repository provides a workflow for training and evaluating machine learning models on microbial abundance time series. Its aim is to predict changes in microbial communities and identify observations that fall outside model-derived prediction intervals.

The workflow starts with a microbial abundance table and a reference list of taxa. An optional matching metadata table can describe environmental conditions or other sample characteristics. Metadata are used as input features; prediction targets are microbial abundances. Outputs include trained models, predictions, evaluation metrics, prediction intervals, and an interactive plot for exploring potential anomalies.

The workflow is under active development. Read [Project status](#project-status) and [Current execution limitations](#current-execution-limitations) before running an analysis. The input examples illustrate file structure; they are not a validated end-to-end training example.

## Workflow overview

| Step | What happens | Main result |
| --- | --- | --- |
| Prepare inputs | Supply abundance data, a reference taxa list, and matching metadata when enabled. | Two or three input files |
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

The supplied environment contains Linux-specific package builds. Installation on macOS or Windows has not been established. The commands in this README assume that you launch them from the repository's main directory. See [File paths](#file-paths) when using another working directory.

## Prepare your input data

By default, the workflow looks for these files:

```text
data/
├── timeseries.tsv
├── metadata.tsv
└── taxa.tsv
```

You can use different locations and filenames by editing `config/profile.yaml`.

`metadata.tsv` is only read when metadata are enabled. The `tcn_lstm` and `lstm` architectures can run without it; `attention` and `metadata_parallel` require metadata.

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

Use a tab-separated table with metadata variables in rows and sampling timepoints in columns. The sample identifiers must match the abundance table exactly. Column order may differ: metadata are aligned by identifier before both tables are sorted chronologically. Missing, extra, or duplicate sample identifiers are rejected.

<!-- markdownlint-disable MD010 -->

```tsv
variable	2025-01-01	2025-01-02	2025-01-03
temperature	18.2	18.5	19.1
pH	7.1	7.0	7.2
```

<!-- markdownlint-enable MD010 -->

For example, `temperature` and `pH` describe the conditions at each sampling timepoint. Use finite numeric values for ordinary model inputs. Fields listed in `columns_to_drop` in `config/encodings.yaml` are excluded before numeric validation.

Fields named `month` and `season` are converted to `month_ord` and `season_ord` using the mappings in `config/encodings.yaml`. Names are case-insensitive and surrounding whitespace is ignored; numeric codes already present in the mappings are also accepted. Unknown category values and unsupported text in other retained fields produce an error identifying the field.

Metadata field names must be unique and nonempty. `Time` and names matching `Target<number>` are reserved. Do not supply both a source category and its generated field, such as `season` and `season_ord`.

### Reference taxa: `taxa.tsv`

Provide one taxonomy label per line, with no header or abundance columns:

```text
d__Bacteria; p__Bacteroidota; c__Bacteroidia; o__Bacteroidales; f__Bacteroidaceae; g__Bacteroides
d__Bacteria; p__Bacillota; c__Clostridia; o__Lachnospirales; f__Lachnospiraceae; g__Blautia
```

The labels must match the genus-level labels used in the abundance table after preprocessing. This file defines the model's taxa and their order. Keep the same reference list when applying or retraining an existing model on another dataset, including taxa absent from that dataset.

### Check the inputs before analysis

- Use actual tab characters to separate abundance and metadata columns.
- Use identical, unique timepoint identifiers in the abundance and metadata tables; column order may differ.
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
  include_metadata: True    # Include metadata as input features.
  number_ensemble_model: 50 # Number of models used to estimate prediction intervals.

model_architecture: tcn_lstm
```

`model_architecture` selects the model to train. The code contains options for `tcn_lstm` (a temporal convolutional network with an LSTM residual model), `lstm`, `attention`, and `metadata_parallel`. These options use different preprocessing and model paths; their end-to-end compatibility still needs validation.

Set `parameters.include_metadata` to `True` or `False` in the profile. For standalone preprocessing, use `--include-metadata True` or `--include-metadata False` (case-insensitive). Standalone preprocessing defaults to `False`; the supplied profile enables metadata.

For `tcn_lstm` and `lstm`, enabled metadata are added to the input features. For `attention` and `metadata_parallel`, metadata are required and supplied as a separate input branch; they are kept out of the bacterial lag features. All architectures use microbial abundances as prediction targets. Requesting a metadata-specific architecture with metadata disabled produces a command-line error.

After changing metadata fields or enabling/disabling metadata, rerun preprocessing and train new models in a new experiment directory. Existing splits, scalers, and models may have incompatible feature or target dimensions.

The `paths` section names output subdirectories such as `models`, `tables`, `intermediate`, and `figures`. Most readers can keep these names unchanged. With the example above, the runner places experiment outputs under `results/my_experiment/`. The supplied profile instead uses the experiment name `tcn_residual_experiment_01`.

The profile also contains training and validation proportions and a retraining setting. These are not currently connected to the runner's stage commands; see the limitations below before changing them.

### File paths

The runner locates its default `config/profile.yaml` and stage scripts relative to the repository, and launches each stage with the same Python interpreter used to launch the runner. Built-in metadata encodings and fallback model configuration are also located relative to the repository.

Relative input and output paths are resolved from the directory where you launch the command. A relative `--profile` path is also resolved from that directory; selecting a different profile does not make data paths relative to the profile's directory. Absolute paths can be used for inputs, outputs, and the profile.

The runner passes experiment-specific paths to all stages, including the reference taxa file for interval calculation and the prepared table and taxa mapping for visualization. Parent directories are created before output files are written. Logging continues to use the repository's `logs/` directory.

Scaler arguments always name the **complete file**, such as `results/my_experiment/models/scaler.pkl`. Pass that same filename when saving, evaluating, or calculating intervals; do not pass just the model directory. Existing scaler files do not need to be moved. For standalone scripts, use `--scaler-path`; the interval script accepts both `--scaler` and `--scaler-path`. The training script's `--path` argument continues to name a model directory.

Paths with spaces are supported. When entering them in a terminal, quote the entire argument, for example `--scaler-path "results/my experiment/models/scaler.pkl"`. The runner passes paths directly to subprocesses as separate arguments.

## Run pipeline stages

Use `scripts/run_pipeline.py` to build stage commands from the profile. With no stage selection, the runner executes all six standard stages in order:

```bash
python scripts/run_pipeline.py
```

To run only preprocessing:

```bash
python scripts/run_pipeline.py --stages preprocess
```

To select several stages, separate their names with commas:

```bash
python scripts/run_pipeline.py --stages preprocess,train,predict,evaluate
```

The implemented stages are `preprocess`, `train`, `predict`, `evaluate`, `interval`, and `visualize`. They execute in that fixed order, regardless of their order in `--stages`. Empty selections, unknown stage names, and retraining requests produce a command-line error before output directories are created. Later stages depend on files created by earlier stages, so selecting `predict` alone requires existing models and data splits. Visualization also needs the interval and anomaly outputs.

To inspect the generated commands without executing the stages:

```bash
python scripts/run_pipeline.py --dry-run
```

A dry run loads and validates the profile, then prints each selected stage command once without executing stages or creating output directories. Combine `--dry-run` with `--stages` to preview a subset. It does not verify input contents, model compatibility, or successful execution. Use `--profile path/to/profile.yaml` to supply another experiment profile.

### Current execution limitations

The following issues need attention before treating this as a verified end-to-end workflow:

- **Split settings:** the runner does not pass the profile's `training_percentage` and `validation_percentage` to preprocessing. The script's defaults currently apply instead.
- **Retraining:** the runner rejects `retrain` and `retraining` stage requests because retraining uses the separate entrypoint described below. The profile's `retraining` setting does not trigger retraining. Standalone retraining supports `tcn_lstm` and `lstm`; metadata-specific architectures are not yet supported.
- **Forecasting setup:** the intended task is prediction over time. The standard splitting path reshapes features into three positions without explicitly constructing lags from preceding observations; the attention preprocessing path constructs lagged features separately. Input lags and future targets need verification across architectures before interpreting outputs as next-timepoint forecasts.

## Understand the results

For the supplied profile, the runner uses `results/tcn_residual_experiment_01/` as the experiment directory. The table below gives paths relative to that directory and describes the outputs of successful stages.

| Output | What it contains | How to use it |
| --- | --- | --- |
| `tables/complete_df.csv` | Prepared observations with a `Time` column and numbered taxa columns such as `Target1`. | Check that dates and abundance values were assembled as expected. |
| `intermediate/dic_TargTax.pkl` | A mapping from numbered target columns to taxonomy labels. | Connect model targets back to taxa. |
| `intermediate/splits.npz` | Arrays used for training, validation, and testing. | Inputs to model training and prediction. |
| `intermediate/split_sizes.pkl` | Training end and test start positions, saved for every architecture. The legacy key `val_size` is the cumulative training-plus-validation boundary. | Check that plotted predictions use the original split dates. |
| `models/` | Saved model files and the fitted scaler. | Reuse the models and the same numerical scaling. |
| `intermediate/predictions.npz` | Predictions for the training, validation, and test sets. | Inputs to evaluation and plotting. |
| `tables/evaluation_metrics.tsv` | Training and test MAE, RMSE, and NRMSE. | Assess prediction errors; lower values indicate smaller errors. |
| `tables/prediction_interval.tsv` | Per-taxon interval bounds and mean predictions, with the original test sample labels in `Time` and a test-relative `timepoint` index. | Examine the range estimated by the ensemble at each test observation. |
| `tables/prediction_interval_anomalies.tsv` | Observations outside the interval, including the taxon, time, observed value, bounds, and direction of the violation. | Identify observations for further investigation. |
| `figures/plot_taxa_anomalies.html` | An interactive Plotly view with a taxon selector. | Open in a web browser to explore the results visually. |

Start with `complete_df.csv` to check the prepared data, then inspect the evaluation table and interactive plot.

Mean absolute error (MAE) measures average absolute prediction error. Root mean squared error (RMSE) gives greater weight to larger errors. Normalized RMSE (NRMSE) divides RMSE by the standard deviation of the observed values in the evaluated set. MAE and RMSE are computed after inverse scaling, so their units depend on the abundance scale supplied as input.

The ensemble produces lower and upper prediction bounds for each taxon. An anomaly is an observation above the upper bound or below the lower bound in the test period. This flags a departure from the model-derived range; it does not by itself establish a biological community shift or its cause. Interpretation depends on input quality, model performance, and the interval method's assumptions and coverage.

### Explore the interactive plot

The plot shows observations, training predictions, validation predictions, test predictions, and any available interval bounds in the original abundance units. Use the taxon dropdown to switch both the curves and anomaly table. When no anomalies are detected, the output TSV retains its headers and the plot displays a "No anomalies" table.

The runner supplies the prepared table, taxa mapping, predictions, scaler, split boundaries, and interval outputs to the plotter. For a standalone call, supply paths from the same experiment:

```bash
python scripts/plot_taxa_anomalies.py \
  --complete results/my_experiment/tables/complete_df.csv \
  --dic-taxa results/my_experiment/intermediate/dic_TargTax.pkl \
  --predictions results/my_experiment/intermediate/predictions.npz \
  --scaler-path results/my_experiment/models/scaler.pkl \
  --split-sizes results/my_experiment/intermediate/split_sizes.pkl \
  --prediction-interval results/my_experiment/tables/prediction_interval.tsv \
  --anomalies results/my_experiment/tables/prediction_interval_anomalies.tsv \
  --output results/my_experiment/figures/plot_taxa_anomalies.html
```

The scaler is required to restore saved predictions to abundance units. `--split-sizes` is optional for standalone plotting because split boundaries can be inferred from prediction array lengths; when provided, the saved boundaries are checked against those lengths. Prediction rows must cover the prepared observations exactly.

New interval tables carry explicit `Time` labels. Legacy interval tables without `Time` use the test boundary from predictions or the split-size file. Only surplus leading NaN padding is removed; gaps within the test period keep their original positions. Mismatched lengths or sample labels produce an error instead of shifting curves to the end of the time series. Regenerate inconsistent interval outputs before plotting them.

## Retrain existing models on a new dataset

Use `scripts/retraining.py` to adapt saved `tcn_lstm` or standalone `lstm` models. Retraining reuses the original fitted scaler, reference taxa, and feature schema. It transforms the new data without fitting a new scaler and writes new models and results into a **new or empty output directory**. Source models, scaler, mapping, and input files are left unchanged.

For a TCN with a residual LSTM, for example:

```bash
python scripts/retraining.py \
  --timeseries data/new_site/timeseries.tsv \
  --taxa data/reference_taxa.tsv \
  --metadata data/new_site/metadata.tsv \
  --include-metadata \
  --model-architecture tcn_lstm \
  --tcn-path results/source_experiment/models/tcn_model.h5 \
  --lstm-path results/source_experiment/models/lstm_model.h5 \
  --scaler-path results/source_experiment/models/scaler.pkl \
  --source-mapping results/source_experiment/intermediate/dic_TargTax.pkl \
  --output-dir results/retrained_site \
  --train-percentage 0.5 \
  --val-percentage 0.1 \
  --epochs 10 \
  --num-models 5
```

For a standalone LSTM, select `--model-architecture lstm` and omit `--tcn-path`. The source models must include their saved training configuration and accept the same input shape and microbial targets as the current preprocessing. Older models with incompatible input or output shapes are rejected.

`--include-metadata` is a Boolean flag for this standalone command. Omit it and `--metadata` if the original model used no metadata. When metadata are enabled, their retained fields must match the source scaler's named features; field order is aligned to that schema. The reference taxa and their order must match `--source-mapping`. Taxa absent from the new abundance table remain zero-filled against that fixed reference.

Training and validation fractions must be positive, sum to less than one, and leave nonempty training, validation, and test sets. `--epochs` (default 10) and `--batch-size` (default 32) apply to both the primary retrained models and each interval ensemble member. `--num-models` (default 5) controls an additional ensemble of independently retrained copies of the source models. Validation residuals determine interval widths in abundance units; test observations are reserved for evaluation and anomaly detection. The interval method's statistical coverage still needs scientific verification.

The output directory contains the usual `models/`, `tables/`, `intermediate/`, and `figures/` artifacts: retrained model files, an unchanged copy of the original scaler, prepared observations, taxa mapping, splits and boundaries, predictions, metrics, dated intervals, anomalies, and an interactive plot. Each retraining run needs a fresh output directory so that an earlier run is not overwritten.

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
| `scripts/retraining.py` | Adapt saved `tcn_lstm` or `lstm` models to new data using the original scaler and schema, saving a separate experiment. |

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
