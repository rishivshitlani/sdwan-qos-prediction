"""Train baseline regression models on the processed Zenodo 5G QoS dataset.

This wrapper keeps the Zenodo modelling setup explicit: the user can select the
target, while derived/outcome columns that would leak the answer are dropped
from model inputs by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from process_zenodo_dataset import DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_PATH, process_zenodo_dataset
from train_baseline import train_baselines


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Keep all default paths inside the existing project structure so the command
# can be run without passing arguments during normal experimentation.
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "reports" / "model_results" / "zenodo_baseline_results.csv"
RECOMMENDED_BANDWIDTH_RESULTS_PATH = (
    PROJECT_ROOT / "reports" / "model_results" / "zenodo_baseline_recommended_bandwidth_results.csv"
)
DEFAULT_TARGET_COLUMN = "actual_throughput_mbps"
SUPPORTED_TARGET_COLUMNS = [
    "actual_throughput_mbps",
    "recommended_bandwidth_percent",
]
DATASET_NAME_BY_TARGET = {
    "actual_throughput_mbps": "zenodo_13754300_actual_throughput",
    "recommended_bandwidth_percent": "zenodo_13754300_recommended_bandwidth",
}

# These columns are measured outcomes, derived outcomes, or trace-only metadata.
# The selected target is removed from this list below so it can still be used
# as y, while the other columns are excluded from X.
DEFAULT_DROP_COLUMNS = [
    "recommended_bandwidth_percent",
    "jitter_ms",
    "packet_loss_percent",
    "actual_throughput_mbps",
    "source_file",
    "protocol",
    "application_type",
    "link_type",
    "sdr",
]


def ensure_processed_dataset(args: argparse.Namespace) -> None:
    """Create the processed Zenodo dataset unless the user asks to reuse it."""
    # By default, regenerate the processed CSV so training reflects the latest
    # raw Zenodo files and the latest feature-engineering code.
    if args.reuse_processed and args.input_path.exists():
        print(f"Using existing processed dataset: {args.input_path}")
        return

    # process_zenodo_dataset reads raw Zenodo CSV files and writes one clean,
    # modelling-ready table with SD-WAN-style feature names.
    process_zenodo_dataset(
        input_dir=args.raw_input_dir,
        output_path=args.input_path,
        assumed_link_capacity_mbps=args.assumed_link_capacity_mbps,
        skip_owd=args.skip_owd,
        chunksize=args.chunksize,
    )


def parse_args() -> argparse.Namespace:
    """Read command-line options for Zenodo baseline training."""
    parser = argparse.ArgumentParser(
        description="Process Zenodo 5G QoS data and train baseline throughput models."
    )
    parser.add_argument("--raw-input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--input-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Where to write model metrics. Defaults depend on the selected target.",
    )
    parser.add_argument(
        "--target-column",
        "--target",
        choices=SUPPORTED_TARGET_COLUMNS,
        default=DEFAULT_TARGET_COLUMN,
        help=(
            "Zenodo target to predict. Use actual_throughput_mbps for throughput "
            "prediction, or recommended_bandwidth_percent for bandwidth-allocation prediction."
        ),
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Dataset label to store in the results CSV. Defaults depend on the selected target.",
    )
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of k-fold cross-validation folds to run on the training split. Use 0 or 1 to disable.",
    )
    parser.add_argument("--assumed-link-capacity-mbps", type=float, default=150.0)
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument(
        "--skip-owd",
        action="store_true",
        # OWD aggregation is now the default so latency, packet_count, and
        # flow_duration_sec are real aggregated features. This flag keeps a
        # quick mode available for fast debugging.
        help="Skip OWD packet aggregation for a faster throughput-only run.",
    )
    parser.add_argument(
        "--include-owd",
        action="store_true",
        help="Deprecated no-op: OWD aggregation is included by default.",
    )
    parser.add_argument(
        "--reuse-processed",
        action="store_true",
        help="Use --input-path if it already exists instead of regenerating it.",
    )
    parser.add_argument(
        "--drop-column",
        action="append",
        default=[],
        help="Additional input column to exclude from model training. Can be used more than once.",
    )
    args = parser.parse_args()

    if args.include_owd:
        print("Note: --include-owd is now the default; use --skip-owd only for quick runs.")

    args.dataset_name = args.dataset_name or DATASET_NAME_BY_TARGET[args.target_column]

    # Keep the original actual-throughput output path for the default run, but
    # write recommended-bandwidth experiments to a separate file by default.
    if args.output_path is None:
        if args.target_column == "recommended_bandwidth_percent":
            args.output_path = RECOMMENDED_BANDWIDTH_RESULTS_PATH
        else:
            args.output_path = DEFAULT_RESULTS_PATH

    return args


if __name__ == "__main__":
    args = parse_args()

    # Step 1: Build the processed Zenodo dataset, or reuse it if requested.
    ensure_processed_dataset(args)

    # Step 2: Remove outcome/leakage columns from X before model training.
    # Example: recommended_bandwidth_percent is derived from throughput, and
    # actual_throughput_mbps is part of that derivation. They must not be used
    # as inputs for each other.
    drop_columns = DEFAULT_DROP_COLUMNS + args.drop_column

    # Step 3: If the target is listed in the default drops, keep it available as
    # y. The generic trainer removes the target from X separately.
    if args.target_column in drop_columns:
        drop_columns = [column for column in drop_columns if column != args.target_column]

    # Step 4: Train the generic baseline models and write metrics to reports/.
    results = train_baselines(
        input_path=args.input_path,
        output_path=args.output_path,
        target_column=args.target_column,
        dataset_name=args.dataset_name,
        test_size=args.test_size,
        random_state=args.random_state,
        drop_columns=drop_columns,
        cv_folds=args.cv_folds,
    )

    print(f"\nZenodo baseline results written to: {args.output_path}")
    print(results.to_string(index=False))
