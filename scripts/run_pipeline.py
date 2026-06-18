#!/usr/bin/env python3
"""Run the full pipeline using a profile YAML.

This script loads `config/profile.yaml` (or a provided profile path),
validates it, then runs the standard stages in order by invoking the
existing CLI scripts in `scripts/` with arguments derived from the profile.

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
from src.utils.config import load_profile, validate_profile

VALID_REQUESTS = {"preprocess","train","predict","evaluate","visualize","retrain"}

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
    logging.info("Running: %s", cmd)
    if dry_run:
        print(cmd)
        return 0
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
    p["timeseries"] = os.path.join(input_dir, input_files.get("timeseries", "timeseries.tsv"))
    p["taxa"] = os.path.join(input_dir, input_files.get("taxa", "taxa.tsv"))
    p["metadata"] = os.path.join(input_dir, input_files.get("metadata", "metadata.tsv"))

    # output/derived files
    paths = profile.get("paths", {})
    p["complete_csv"] = os.path.join(base, paths.get("tables", "tables"), "complete_df.csv")
    p["mapping_output"] = os.path.join(base, paths.get("intermediate", "intermediate"), "dic_TargTax.pkl")
    p["splits_output"] = os.path.join(base, paths.get("intermediate", "intermediate"), "splits.npz")
    p["split_sizes"] = os.path.join(base, paths.get("intermediate", "intermediate"), "split_sizes.pkl")
    p["predictions_npz"] = os.path.join(base, paths.get("intermediate", "intermediate"), "predictions.npz")
    p["tcn_model"] = os.path.join(base, paths.get("models", "models"), "tcn_model.h5")
    p["lstm_model"] = os.path.join(base, paths.get("models", "models"), "lstm_model.h5")
    p["evaluation_output"] = os.path.join(base, paths.get("tables", "tables"), "evaluation_metrics.tsv")
    return p


def main():
    parser = argparse.ArgumentParser(description="Run full pipeline using profile YAML")
    parser.add_argument("--profile", type=str, default="config/profile.yaml", help="Path to profile YAML")
    parser.add_argument("--stages", type=str, default=None, help="Comma-separated stages to run (preprocess,train,predict,evaluate,visualize)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    profile = load_profile(args.profile)
    validate_profile(profile, required_keys=["project", "paths"]) 
    _ensure_dirs(profile)
    pp = build_paths(profile)

    requested = None
    if args.stages:
        requested = [s.strip() for s in args.stages.split(",") if s.strip()]

    if requested is None:
        raise ValueError("requested cannot be None")

    if not isinstance(requested, list):
        raise TypeError(f"requested must be a list, got {type(requested).__name__}")

    if len(requested) == 0:
        raise ValueError("requested cannot be empty")

    invalid = [r for r in requested if r not in VALID_REQUESTS]

    if invalid:
        raise ValueError(
            f"Invalid requested values: {invalid}. "
            f"Valid options: {VALID_REQUESTS}"
        )

    # Stage: preprocessing
    if (requested is None) or ("preprocess" in requested):
        cmd = (
            f"python scripts/preprocessing.py --timeseries {pp['timeseries']} --metadata {pp['metadata']} "
            f"--taxa {pp['taxa']} --include-metadata {str(profile.get('parameters', {}).get('include_metadata', False))} "
            f"--output {pp['complete_csv']} --mapping-output {pp['mapping_output']} --splits-output {pp['splits_output']} --splits-sizes {pp['split_sizes']}"
        )
        run_cmd(cmd, dry_run=args.dry_run)
        print(cmd)

    # Stage: training
    if (requested is None) or ("train" in requested):
        cmd = (
            f"python scripts/training.py --splits-input {pp['splits_output']} --tcn-path {pp['tcn_model']} --lstm-path {pp['lstm_model']}"
        )
        run_cmd(cmd, dry_run=args.dry_run)

    # Stage: prediction
    if (requested is None) or ("predict" in requested):
        cmd = (
            f"python scripts/prediction.py --splits-input {pp['splits_output']} --tcn-path {pp['tcn_model']} --lstm-path {pp['lstm_model']} --output {pp['predictions_npz']}"
        )
        run_cmd(cmd, dry_run=args.dry_run)

    # Stage: evaluation
    if (requested is None) or ("evaluate" in requested):
        cmd = (
            f"python scripts/evaluation.py --prediction-results {pp['predictions_npz']} --splits {pp['splits_output']} --output {pp['evaluation_output']}"
        )
        run_cmd(cmd, dry_run=args.dry_run)

    # Stage: visualize (optional)
    if (requested is None) or ("visualize" in requested):
        # call the taxa violation plot if it exists
        project_base = _get_project_base(profile)
        plot_out = os.path.join(project_base, profile.get("paths", {}).get("figures", "figures"), "plot_taxa_anomalies.html")
        prediction_interval_path = os.path.join(project_base, profile.get("paths", {}).get("tables", "tables"), "prediction_interval.tsv")
        cmd = (
            f"python scripts/plot_taxa_anomalies.py --prediction-interval {prediction_interval_path} "
            f"--predictions {pp['predictions_npz']} --split-sizes {pp['split_sizes']} --output {plot_out}"
        )
        run_cmd(cmd, dry_run=args.dry_run)

    logging.info("Pipeline finished successfully")


if __name__ == "__main__":
    main()
