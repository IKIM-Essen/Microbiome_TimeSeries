#!/usr/bin/env python3
"""Run the full pipeline using a profile YAML.

This script loads `config/profile.yaml` (or a provided profile path) and
validates it, then runs all standard stages by default, or a selected subset,
in pipeline order. It invokes the existing CLI scripts in `scripts/` with
arguments derived from the profile.

It prefers to call the CLI scripts as subprocesses so existing behavior
and logging remain unchanged.
"""
import argparse
import logging
import os
import shlex
import subprocess
import sys

# allow imports from repo
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

DEFAULT_STAGES = (
    "preprocess",
    "train",
    "predict",
    "evaluate",
    "interval",
    "visualize",
)


def _parse_stages(value):
    requested = tuple(stage.strip() for stage in value.split(","))
    if any(not stage for stage in requested):
        raise argparse.ArgumentTypeError(
            "Stage selection must be a nonempty comma-separated list "
            "without empty stage names."
        )

    if any(stage in {"retrain", "retraining"} for stage in requested):
        raise argparse.ArgumentTypeError(
            "Retraining is not implemented by this pipeline runner. "
            "The separate scripts/retraining.py workflow is still under development."
        )

    invalid = [stage for stage in requested if stage not in DEFAULT_STAGES]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Unknown stage(s): {', '.join(invalid)}. "
            f"Choose from: {', '.join(DEFAULT_STAGES)}."
        )
    return requested


def _get_project_base(profile):
    base_dir = profile.get("project", {}).get("base_dir", "results")
    project_name = profile.get("project", {}).get("name")
    if project_name:
        return os.path.join(base_dir, project_name)
    return base_dir


def _ensure_dirs(profile):
    # Create base output directories referenced in the profile
    base = _get_project_base(profile)
    paths = profile.get("paths", {})
    for p in paths.values():
        d = os.path.join(base, p)
        os.makedirs(d, exist_ok=True)


def run_cmd(cmd, dry_run=False):
    if dry_run:
        print(cmd)
        return 0
    logging.info("Running: %s", cmd)
    try:
        res = subprocess.run(shlex.split(cmd), check=True)
        return res.returncode
    except subprocess.CalledProcessError as e:
        logging.error("Command failed: %s", e)
        raise


def build_paths(profile):
    base = _get_project_base(profile)
    input_dir = profile.get("project", {}).get("input_dir", "data")
    p = {}
    # input files
    input_files = profile.get("input_files", {})
    p["timeseries"] = os.path.join(
        input_dir, input_files.get("timeseries", "timeseries.tsv")
    )
    p["taxa"] = os.path.join(input_dir, input_files.get("taxa", "taxa.tsv"))
    p["metadata"] = os.path.join(input_dir, input_files.get("metadata", "metadata.tsv"))

    # output/derived files
    paths = profile.get("paths", {})
    p["complete_csv"] = os.path.join(
        base, paths.get("tables", "tables"), "complete_df.csv"
    )
    p["mapping_output"] = os.path.join(
        base, paths.get("intermediate", "intermediate"), "dic_TargTax.pkl"
    )
    p["splits_output"] = os.path.join(
        base, paths.get("intermediate", "intermediate"), "splits.npz"
    )
    p["split_sizes"] = os.path.join(
        base, paths.get("intermediate", "intermediate"), "split_sizes.pkl"
    )
    p["predictions_npz"] = os.path.join(
        base, paths.get("intermediate", "intermediate"), "predictions.npz"
    )
    p["model_dir"] = os.path.join(base, paths.get("models", "models"))
    p["tcn_model"] = os.path.join(base, paths.get("models", "models"), "tcn_model.h5")
    p["lstm_model"] = os.path.join(base, paths.get("models", "models"), "lstm_model.h5")
    p["attention_model"] = os.path.join(
        base, paths.get("models", "models"), "attention_model.h5"
    )
    p["metadata_model"] = os.path.join(
        base, paths.get("models", "models"), "meta_lstm.h5"
    )
    p["evaluation_output"] = os.path.join(
        base, paths.get("tables", "tables"), "evaluation_metrics.tsv"
    )
    return p


def main():
    parser = argparse.ArgumentParser(description="Run full pipeline using profile YAML")
    parser.add_argument(
        "--profile",
        type=str,
        default="config/profile.yaml",
        help="Path to profile YAML",
    )
    parser.add_argument(
        "--stages",
        type=_parse_stages,
        default=DEFAULT_STAGES,
        metavar="STAGE[,STAGE...]",
        help=(
            f"Comma-separated stages: {', '.join(DEFAULT_STAGES)}. "
            "Defaults to all stages, executed in pipeline order. "
            "Retraining is not supported by this runner."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing stages or creating output directories",
    )
    args = parser.parse_args()
    requested = args.stages

    # Parse CLI arguments before loading the pipeline's dependencies.
    from src.utils.config import load_profile, validate_profile

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    profile = load_profile(args.profile)
    validate_profile(profile, required_keys=["project", "paths"])
    pp = build_paths(profile)
    if not args.dry_run:
        _ensure_dirs(profile)

    # Stage: preprocessing
    if "preprocess" in requested:
        cmd = (
            f"python scripts/preprocessing.py --timeseries {pp['timeseries']} --metadata {pp['metadata']} "
            f"--taxa {pp['taxa']} --include-metadata {profile.get('parameters', {}).get('include_metadata', False)} "
            f"--output {pp['complete_csv']} --mapping-output {pp['mapping_output']} --model-architecture {profile.get('model_architecture')} "
            f"--splits-output {pp['splits_output']} --scaler-path {pp['model_dir']} --splits-sizes {pp['split_sizes']} "
        )
        run_cmd(cmd, dry_run=args.dry_run)

    # Stage: training
    if "train" in requested:
        model_arch = profile.get("model_architecture")
        model_dir = pp["model_dir"]
        cmd = f"python scripts/training.py --splits-input {pp['splits_output']} --path {model_dir}"
        if model_arch:
            cmd = cmd + f" --model-architecture {model_arch}"
        run_cmd(cmd, dry_run=args.dry_run)

    # Stage: prediction
    if "predict" in requested:
        model_arch = profile.get("model_architecture")
        cmd = f"python scripts/prediction.py --splits-input {pp['splits_output']} --tcn-path {pp['tcn_model']} --lstm-path {pp['lstm_model']} --output {pp['predictions_npz']}"
        if model_arch:
            cmd = cmd + f" --model-architecture {model_arch}"
        if model_arch == "metadata_parallel":
            cmd = cmd + f" --attention-path {pp['metadata_model']}"
        if model_arch == "attention":
            cmd = cmd + f" --attention-path {pp['attention_model']}"
        run_cmd(cmd, dry_run=args.dry_run)

    # Stage: evaluation
    if "evaluate" in requested:
        model_arch = profile.get("model_architecture")
        cmd = f"python scripts/evaluation.py --prediction-results {pp['predictions_npz']} --splits {pp['splits_output']} --output {pp['evaluation_output']} --scaler-path {pp['model_dir']}"
        if model_arch:
            cmd = cmd + f" --model-architecture {model_arch}"
        run_cmd(cmd, dry_run=args.dry_run)

    if "interval" in requested:
        # Build sensible defaults for interval stage using profile and built paths
        model_arch = profile.get("model_architecture")
        num_models = profile.get("parameters", {}).get("number_ensemble_model", 5)
        base = _get_project_base(profile)
        tables_dir = os.path.join(
            base, profile.get("paths", {}).get("tables", "tables")
        )
        interval_output = os.path.join(tables_dir, "prediction_interval.tsv")
        anomalies_output = os.path.join(tables_dir, "prediction_interval_anomalies.tsv")

        scaler_path = os.path.join(pp["model_dir"])

        cmd = (
            f"python scripts/interval.py --num-models {num_models} --splits-input {pp['splits_output']} "
            f"--scaler {scaler_path} --tcn-path {pp['tcn_model']} --lstm-path {pp['lstm_model']} "
            f"--output {interval_output} --complete-input {pp['complete_csv']} --dic-taxa {pp['mapping_output']} "
            f"--anomalies-output {anomalies_output}  --model-architecture {model_arch}"
        )
        run_cmd(cmd, dry_run=args.dry_run)

    # Stage: visualize (optional)
    if "visualize" in requested:
        # call the taxa violation plot if it exists
        project_base = _get_project_base(profile)
        base = _get_project_base(profile)
        tables_dir = os.path.join(
            base, profile.get("paths", {}).get("tables", "tables")
        )
        interval_output = os.path.join(tables_dir, "prediction_interval.tsv")
        anomalies_output = os.path.join(tables_dir, "prediction_interval_anomalies.tsv")
        plot_out = os.path.join(
            project_base,
            profile.get("paths", {}).get("figures", "figures"),
            "plot_taxa_anomalies.html",
        )
        prediction_interval_path = os.path.join(
            project_base,
            profile.get("paths", {}).get("tables", "tables"),
            "prediction_interval.tsv",
        )
        cmd = (
            f"python scripts/plot_taxa_anomalies.py --prediction-interval {prediction_interval_path} "
            f"--predictions {pp['predictions_npz']} --split-sizes {pp['split_sizes']} --anomalies {anomalies_output} --prediction-interval {interval_output} --output {plot_out}"
        )
        run_cmd(cmd, dry_run=args.dry_run)

    if args.dry_run:
        logging.info("Dry run finished; no stages were executed")
    else:
        logging.info("Pipeline finished successfully")


if __name__ == "__main__":
    main()
