"""Clean public tabular datasets and report label imbalance.

This script is the second step after ``process_public_dataset.py``. It profiles
each dataset under ``data/raw``, identifies constant-zero feature columns,
checks whether the label distribution is imbalanced, then writes cleaned CSV
files under ``data/processed/cleaned``.

Each top-level folder under ``data/raw`` is treated as a separate dataset. This
keeps CICIDS, MAWI, UNIBS, Kaggle exports, and other future sources separate so
model performance can be compared per dataset instead of mixing schemas.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import re

import numpy as np
import pandas as pd

from process_public_dataset import (
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT_DIR,
    clean_columns,
    clean_label_values,
    detect_label_column,
    find_dataset_files,
    format_path_for_display,
    group_dataset_files,
    read_dataset_file,
    safe_sheet_name,
)


DEFAULT_CLEANED_DIR = DEFAULT_OUTPUT_DIR / "cleaned"
MISSING_POLICY_CHOICES = ("all", "label", "required", "none")


def parse_required_columns(required_columns: str | None) -> list[str]:
    """Split a comma-separated column list from the command line."""
    if not required_columns:
        return []
    return [column.strip() for column in required_columns.split(",") if column.strip()]


def update_column_profile(
    column_profile: dict[str, dict[str, object]],
    chunk: pd.DataFrame,
    column_order: list[str],
) -> None:
    """Collect enough information to detect constant-zero columns."""
    for column in chunk.columns:
        # Keep the first-seen column order so the cleaned file remains easy to
        # compare with the original raw dataset.
        if column not in column_order:
            column_order.append(column)

        series = chunk[column]
        profile = column_profile.setdefault(
            column,
            {
                "non_null_count": 0,
                "missing_count": 0,
                "infinite_count": 0,
                "numeric_count": 0,
                "min": np.nan,
                "max": np.nan,
            },
        )

        profile["non_null_count"] += int(series.notna().sum())
        profile["missing_count"] += int(series.isna().sum())

        numeric = pd.to_numeric(series, errors="coerce")
        infinite_mask = np.isinf(numeric)
        profile["infinite_count"] += int(infinite_mask.sum())

        clean_numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna()
        if clean_numeric.empty:
            continue

        profile["numeric_count"] += int(clean_numeric.count())
        chunk_min = float(clean_numeric.min())
        chunk_max = float(clean_numeric.max())
        profile["min"] = chunk_min if pd.isna(profile["min"]) else min(float(profile["min"]), chunk_min)
        profile["max"] = chunk_max if pd.isna(profile["max"]) else max(float(profile["max"]), chunk_max)


def profile_dataset(
    dataset_files: list[Path],
    chunksize: int,
    requested_label_column: str | None,
) -> dict[str, object]:
    """First pass: inspect a dataset before writing cleaned output."""
    column_profile: dict[str, dict[str, object]] = {}
    label_counts: Counter[str] = Counter()
    column_order: list[str] = []
    rows_seen = 0
    detected_label_column: str | None = None

    for dataset_file in dataset_files:
        print(f"Profiling {format_path_for_display(dataset_file)}")
        for chunk in read_dataset_file(dataset_file, chunksize=chunksize):
            chunk = clean_columns(chunk)
            update_column_profile(column_profile, chunk, column_order)

            current_label_column = detect_label_column(list(chunk.columns), requested_label_column)
            if current_label_column:
                detected_label_column = detected_label_column or current_label_column
                label_counts.update(clean_label_values(chunk[current_label_column]))

            rows_seen += len(chunk)

    constant_zero_columns = [
        column
        for column, profile in column_profile.items()
        if column != detected_label_column
        and int(profile["numeric_count"]) > 0
        and float(profile["min"]) == 0.0
        and float(profile["max"]) == 0.0
    ]

    return {
        "rows_seen": rows_seen,
        "column_profile": column_profile,
        "column_order": column_order,
        "label_counts": label_counts,
        "label_column": detected_label_column,
        "constant_zero_columns": constant_zero_columns,
    }


def calculate_imbalance_summary(label_counts: Counter[str], threshold: float) -> dict[str, object]:
    """Return class-balance metrics for the detected label column."""
    total_labeled_rows = sum(label_counts.values())
    if total_labeled_rows == 0:
        return {
            "is_imbalanced": "",
            "majority_label": "",
            "majority_count": 0,
            "majority_percent": "",
            "minority_label": "",
            "minority_count": 0,
            "minority_percent": "",
            "imbalance_threshold": threshold,
        }

    sorted_counts = sorted(label_counts.items(), key=lambda item: item[1], reverse=True)
    majority_label, majority_count = sorted_counts[0]
    minority_label, minority_count = sorted_counts[-1]
    majority_percent = majority_count / total_labeled_rows
    minority_percent = minority_count / total_labeled_rows

    return {
        "is_imbalanced": majority_percent >= threshold,
        "majority_label": majority_label,
        "majority_count": majority_count,
        "majority_percent": majority_percent,
        "minority_label": minority_label,
        "minority_count": minority_count,
        "minority_percent": minority_percent,
        "imbalance_threshold": threshold,
    }


def safe_file_stem(name: str) -> str:
    """Create a filesystem-safe name for generated dataset artifacts."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "dataset"


def clean_dataset(
    dataset_name: str,
    dataset_files: list[Path],
    cleaned_dir: Path,
    chunksize: int,
    requested_label_column: str | None,
    imbalance_threshold: float,
    missing_policy: str,
    required_columns: list[str],
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Profile one dataset, remove constant-zero columns, and save cleaned CSV."""
    print(f"\nDataset: {dataset_name}")
    profile = profile_dataset(
        dataset_files=dataset_files,
        chunksize=chunksize,
        requested_label_column=requested_label_column,
    )

    label_counts: Counter[str] = profile["label_counts"]  # type: ignore[assignment]
    label_column = profile["label_column"]
    constant_zero_columns: list[str] = profile["constant_zero_columns"]  # type: ignore[assignment]
    column_order: list[str] = profile["column_order"]  # type: ignore[assignment]

    cleaned_dir.mkdir(parents=True, exist_ok=True)
    output_path = cleaned_dir / f"{safe_file_stem(dataset_name)}_cleaned.csv"

    # Regenerate the cleaned file from scratch on every run. This avoids
    # accidentally appending duplicate rows when the script is rerun.
    if output_path.exists():
        output_path.unlink()

    # Remove only columns proven to be numeric and always zero. Other columns,
    # including the label column, are preserved for downstream modelling.
    output_columns = [column for column in column_order if column not in constant_zero_columns]
    missing_required_columns = [column for column in required_columns if column not in output_columns]
    if missing_policy == "required" and missing_required_columns:
        print(f"Required columns not found in {dataset_name}: {', '.join(missing_required_columns)}")

    rows_written = 0
    rows_dropped_missing = 0
    header_written = False
    cleaned_label_counts: Counter[str] = Counter()

    for dataset_file in dataset_files:
        print(f"Cleaning {format_path_for_display(dataset_file)}")
        for chunk in read_dataset_file(dataset_file, chunksize=chunksize):
            chunk = clean_columns(chunk)

            # Convert infinite values to NaN so a single dropna policy can
            # remove both missing values and invalid numeric infinities.
            chunk = chunk.replace([np.inf, -np.inf], np.nan)
            chunk = chunk.drop(columns=[column for column in constant_zero_columns if column in chunk.columns])
            chunk = chunk.reindex(columns=output_columns)

            before_drop = len(chunk)
            if missing_policy == "all":
                # Strict mode: remove a row if any remaining column is missing.
                # This is useful for clean benchmark files such as CICIDS2017.
                chunk = chunk.dropna()
            elif missing_policy == "label" and label_column and label_column in chunk.columns:
                # Supervised models need a target label, but optional feature
                # fields can remain missing for later imputation or modelling.
                chunk = chunk.dropna(subset=[label_column])
            elif missing_policy == "required":
                # Reusable mode for future datasets: only drop rows missing
                # columns the user marks as necessary for their experiment.
                available_required_columns = [column for column in required_columns if column in chunk.columns]
                if available_required_columns:
                    chunk = chunk.dropna(subset=available_required_columns)
            rows_dropped_missing += before_drop - len(chunk)

            # The imbalance report should describe the rows that remain after
            # cleaning, because those are the rows models will actually use.
            if label_column and label_column in chunk.columns:
                cleaned_label_counts.update(clean_label_values(chunk[label_column]))

            chunk.to_csv(output_path, index=False, mode="a", header=not header_written)
            header_written = True
            rows_written += len(chunk)

    imbalance_summary = calculate_imbalance_summary(cleaned_label_counts, imbalance_threshold)

    summary = {
        "dataset": dataset_name,
        "input_files": len(dataset_files),
        "raw_rows": profile["rows_seen"],
        "cleaned_rows": rows_written,
        "rows_dropped_missing_or_infinite": rows_dropped_missing,
        "missing_policy": missing_policy,
        "required_columns": ", ".join(required_columns),
        "missing_required_columns": ", ".join(missing_required_columns),
        "raw_feature_count": len(column_order),
        "cleaned_feature_count": len(output_columns),
        "constant_zero_features_removed": len(constant_zero_columns),
        "label_column": label_column or "",
        "raw_label_class_count": len(label_counts),
        "cleaned_label_class_count": len(cleaned_label_counts),
        "cleaned_output": str(output_path),
        **imbalance_summary,
    }

    removed_features_df = pd.DataFrame(
        {
            "dataset": dataset_name,
            "removed_constant_zero_feature": constant_zero_columns,
        }
    )
    label_counts_df = pd.DataFrame(
        sorted(cleaned_label_counts.items(), key=lambda item: item[1], reverse=True),
        columns=["label", "count"],
    )
    label_counts_df.insert(0, "dataset", dataset_name)

    print(f"Raw rows: {profile['rows_seen']:,}")
    print(f"Cleaned rows written: {rows_written:,}")
    print(f"Constant-zero features removed: {len(constant_zero_columns)}")
    print(f"Raw label classes: {len(label_counts)}")
    print(f"Cleaned label classes: {len(cleaned_label_counts)}")
    print(f"Imbalanced: {imbalance_summary['is_imbalanced']}")
    print(f"Cleaned output: {output_path}")

    return summary, removed_features_df, label_counts_df


def write_cleaning_report(
    output_path: Path,
    summary_df: pd.DataFrame,
    removed_features_df: pd.DataFrame,
    label_counts_df: pd.DataFrame,
) -> None:
    """Write the cleaning summary to a multi-tab Excel report."""
    used_sheet_names: set[str] = set()
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name=safe_sheet_name("summary", used_sheet_names), index=False)
        removed_features_df.to_excel(
            writer,
            sheet_name=safe_sheet_name("removed_features", used_sheet_names),
            index=False,
        )
        label_counts_df.to_excel(writer, sheet_name=safe_sheet_name("label_counts", used_sheet_names), index=False)


def clean_datasets(
    input_path: Path,
    output_dir: Path,
    chunksize: int,
    label_column: str | None,
    output_prefix: str,
    exclude_patterns: list[str],
    imbalance_threshold: float,
    missing_policy: str,
    required_columns: list[str],
) -> None:
    """Clean all detected datasets and write cleaned data plus reports."""
    dataset_files = find_dataset_files(input_path, exclude_patterns)
    if not dataset_files:
        raise FileNotFoundError(f"No supported dataset files found in {input_path}")

    grouped_files = group_dataset_files(input_path, dataset_files)
    cleaned_dir = output_dir / "cleaned"
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, object]] = []
    removed_feature_reports: list[pd.DataFrame] = []
    label_count_reports: list[pd.DataFrame] = []

    print(f"Cleaning dataset files from: {input_path}")

    for dataset_name, files in sorted(grouped_files.items()):
        # Each group is cleaned independently. The script never merges datasets
        # from different top-level raw folders.
        summary, removed_features_df, label_counts_df = clean_dataset(
            dataset_name=dataset_name,
            dataset_files=files,
            cleaned_dir=cleaned_dir,
            chunksize=chunksize,
            requested_label_column=label_column,
            imbalance_threshold=imbalance_threshold,
            missing_policy=missing_policy,
            required_columns=required_columns,
        )
        summaries.append(summary)
        removed_feature_reports.append(removed_features_df)
        label_count_reports.append(label_counts_df)

    summary_df = pd.DataFrame(summaries)
    removed_features_df = pd.concat(removed_feature_reports, ignore_index=True)
    label_counts_df = pd.concat(label_count_reports, ignore_index=True)

    summary_path = output_dir / f"{output_prefix}_cleaning_summary.csv"
    removed_features_path = output_dir / f"{output_prefix}_removed_constant_zero_features.csv"
    label_counts_path = output_dir / f"{output_prefix}_cleaned_label_counts.csv"
    report_path = output_dir / f"{output_prefix}_cleaning_report.xlsx"

    summary_df.to_csv(summary_path, index=False)
    removed_features_df.to_csv(removed_features_path, index=False)
    label_counts_df.to_csv(label_counts_path, index=False)
    write_cleaning_report(report_path, summary_df, removed_features_df, label_counts_df)

    print(f"\nCleaning summary written to: {summary_path}")
    print(f"Removed-feature report written to: {removed_features_path}")
    print(f"Label-count report written to: {label_counts_path}")
    print(f"Excel report written to: {report_path}")


def parse_args() -> argparse.Namespace:
    """Read command-line options for the cleaning script."""
    parser = argparse.ArgumentParser(
        description="Clean public datasets by removing constant-zero features and reporting label imbalance."
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--output-prefix", default="public_datasets")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Skip files whose path contains this text. Can be used more than once.",
    )
    parser.add_argument(
        "--imbalance-threshold",
        type=float,
        default=0.60,
        help="Mark a dataset imbalanced when the largest class is at least this share of labeled rows.",
    )
    parser.add_argument(
        "--keep-missing-rows",
        action="store_true",
        help="Deprecated shortcut for --missing-policy none.",
    )
    parser.add_argument(
        "--missing-policy",
        choices=MISSING_POLICY_CHOICES,
        default="all",
        help=(
            "Choose which rows to drop after infinite values are converted to missing values: "
            "'all' drops rows missing any column, 'label' drops only missing-label rows, "
            "'required' drops rows missing user-specified required columns, and 'none' keeps them."
        ),
    )
    parser.add_argument(
        "--required-columns",
        default=None,
        help="Comma-separated columns used when --missing-policy required is selected.",
    )
    args = parser.parse_args()
    if args.missing_policy == "required" and not parse_required_columns(args.required_columns):
        parser.error("--missing-policy required needs --required-columns")
    return args


if __name__ == "__main__":
    args = parse_args()
    missing_policy = "none" if args.keep_missing_rows else args.missing_policy
    clean_datasets(
        input_path=args.input_path,
        output_dir=args.output_dir,
        chunksize=args.chunksize,
        label_column=args.label_column,
        output_prefix=args.output_prefix,
        exclude_patterns=args.exclude,
        imbalance_threshold=args.imbalance_threshold,
        missing_policy=missing_policy,
        required_columns=parse_required_columns(args.required_columns),
    )
