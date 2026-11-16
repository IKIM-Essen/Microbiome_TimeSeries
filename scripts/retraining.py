import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(
        description="Retrain standard models on new data while preserving source artifacts."
    )
    parser.add_argument("--timeseries", required=True, help="New abundance TSV")
    parser.add_argument(
        "--taxa",
        required=True,
        help="Original reference taxa list, in the original order",
    )
    parser.add_argument(
        "--metadata", help="Matching metadata TSV when metadata are enabled"
    )
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Include the same metadata fields as the source model",
    )
    parser.add_argument(
        "--model-architecture", choices=("tcn_lstm", "lstm"), default="tcn_lstm"
    )
    parser.add_argument("--tcn-path", help="Source TCN model; required for tcn_lstm")
    parser.add_argument(
        "--lstm-path", required=True, help="Source residual or standalone LSTM model"
    )
    parser.add_argument(
        "--scaler-path", required=True, help="Original fitted scaler file"
    )
    parser.add_argument(
        "--source-mapping", required=True, help="Original intermediate/dic_TargTax.pkl"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="New or empty experiment directory for all retraining outputs",
    )
    parser.add_argument("--train-percentage", type=float, default=0.5)
    parser.add_argument("--val-percentage", type=float, default=0.1)
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Epochs for the primary models and each interval ensemble member",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--num-models",
        type=int,
        default=5,
        help="Number of independently retrained interval ensemble members",
    )
    args = parser.parse_args()
    if args.model_architecture == "tcn_lstm" and args.tcn_path is None:
        parser.error("tcn_lstm requires --tcn-path")
    if args.include_metadata and args.metadata is None:
        parser.error("--include-metadata requires --metadata")

    from src.model_building.retraining_models import retrain_model

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    try:
        paths = retrain_model(
            timeseries=args.timeseries,
            taxa=args.taxa,
            source_mapping=args.source_mapping,
            scaler_path=args.scaler_path,
            output_dir=args.output_dir,
            tcn_path=args.tcn_path,
            lstm_path=args.lstm_path,
            metadata=args.metadata,
            include_metadata=args.include_metadata,
            model_architecture=args.model_architecture,
            train_percentage=args.train_percentage,
            val_percentage=args.val_percentage,
            epochs=args.epochs,
            batch_size=args.batch_size,
            number_models=args.num_models,
        )
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    logging.info("Retraining complete; results are in %s", args.output_dir)
    logging.info("Interactive plot: %s", paths["plot"])


if __name__ == "__main__":
    main()
