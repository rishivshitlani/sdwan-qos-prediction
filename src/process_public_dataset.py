"""Generic tabular dataset inspection script.

Store raw datasets under ``data/raw`` and run this script to understand what
the data looks like before cleaning, feature engineering, or model training.

The script counts rows, files, features, labels, missing values, infinite
values, and simple numeric summaries. It works with common tabular formats such
as CSV, TSV, JSON/JSONL, and Parquet.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
import re

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

SUPPORTED_SUFFIXES = {".csv", ".tsv", ".txt", ".json", ".jsonl", ".ndjson", ".parquet"}
TEXT_ENCODINGS = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
EXCEL_SHEET_MAX_LENGTH = 31
COMMON_LABEL_COLUMNS = [
    "label",
    "Label",
    "LABEL",
    "target",
    "Target",
    "class",
    "Class",
    "category",
    "Category",
]


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove leading/trailing spaces from column names."""
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    return df


def clean_label_values(labels: pd.Series) -> pd.Series:
    """Normalize real label values while leaving missing labels out of counts."""
    cleaned = labels.dropna().astype(str).str.strip().str.replace("\ufffd", "-", regex=False)
    return cleaned[cleaned != ""]


def find_dataset_files(input_path: Path, exclude_patterns: list[str]) -> list[Path]:
    """Return supported dataset files from a file path or directory."""
    input_path = input_path.resolve()

    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported file type: {input_path.suffix}")
        return [input_path]

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    files = [
        path
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    return sorted(path for path in files if not should_exclude(path, exclude_patterns))


def group_dataset_files(input_path: Path, dataset_files: list[Path]) -> dict[str, list[Path]]:
    """Group files into datasets so each dataset can have a separate tab."""
    input_path = input_path.resolve()

    if input_path.is_file():
        return {input_path.stem: dataset_files}

    grouped_files: dict[str, list[Path]] = {}
    for dataset_file in dataset_files:
        relative_path = dataset_file.resolve().relative_to(input_path)
        dataset_name = relative_path.parts[0] if len(relative_path.parts) > 1 else input_path.name
        grouped_files.setdefault(dataset_name, []).append(dataset_file)

    return grouped_files


def should_exclude(path: Path, exclude_patterns: list[str]) -> bool:
    """Return True when any exclude pattern appears in the file path."""
    path_text = str(path)
    return any(pattern and pattern in path_text for pattern in exclude_patterns)


def format_path_for_display(path: Path) -> str:
    """Show repo-relative paths when possible, otherwise show the full path."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def safe_sheet_name(name: str, used_names: set[str]) -> str:
    """Convert a dataset name into a valid, unique Excel sheet name."""
    cleaned = re.sub(r"[\[\]:*?/\\]", "_", name).strip() or "dataset"
    cleaned = cleaned[:EXCEL_SHEET_MAX_LENGTH]
    candidate = cleaned
    suffix = 1

    while candidate in used_names:
        suffix_text = f"_{suffix}"
        candidate = f"{cleaned[:EXCEL_SHEET_MAX_LENGTH - len(suffix_text)]}{suffix_text}"
        suffix += 1

    used_names.add(candidate)
    return candidate


def detect_label_column(columns: list[str], requested_label_column: str | None) -> str | None:
    """Use the requested label column or infer one from common label names."""
    if requested_label_column:
        return requested_label_column if requested_label_column in columns else None

    for candidate in COMMON_LABEL_COLUMNS:
        if candidate in columns:
            return candidate

    lower_to_original = {column.lower(): column for column in columns}
    for candidate in COMMON_LABEL_COLUMNS:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]

    return None


def read_delimited_file(path: Path, chunksize: int, sep: str) -> Iterator[pd.DataFrame]:
    """Read a CSV/TSV file, trying common encodings for messy raw datasets."""
    last_error: Exception | None = None

    for encoding in TEXT_ENCODINGS:
        try:
            yield from pd.read_csv(
                path,
                sep=sep,
                chunksize=chunksize,
                encoding=encoding,
                encoding_errors="replace",
                on_bad_lines="skip",
            )
            return
        except UnicodeDecodeError as error:
            last_error = error
            continue

    if last_error:
        raise last_error


def read_dataset_file(path: Path, chunksize: int) -> Iterator[pd.DataFrame]:
    """Yield dataframe chunks for one supported dataset file."""
    suffix = path.suffix.lower()

    if suffix == ".csv":
        yield from read_delimited_file(path, chunksize=chunksize, sep=",")
    elif suffix in {".tsv", ".txt"}:
        yield from read_delimited_file(path, chunksize=chunksize, sep="\t")
    elif suffix in {".jsonl", ".ndjson"}:
        yield from pd.read_json(path, lines=True, chunksize=chunksize)
    elif suffix == ".json":
        yield pd.read_json(path)
    elif suffix == ".parquet":
        yield pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def update_feature_stats(
    feature_stats: dict[str, dict[str, object]],
    chunk: pd.DataFrame,
) -> None:
    """Update running feature statistics from one dataframe chunk."""
    for column in chunk.columns:
        series = chunk[column]

        # setdefault lets each feature accumulate statistics across all files.
        stats = feature_stats.setdefault(
            column,
            {
                "dtype": str(series.dtype),
                "non_null_count": 0,
                "missing_count": 0,
                "infinite_count": 0,
                "numeric_count": 0,
                "numeric_sum": 0.0,
                "min": np.nan,
                "max": np.nan,
            },
        )

        stats["non_null_count"] += int(series.notna().sum())
        stats["missing_count"] += int(series.isna().sum())

        # Numeric conversion is attempted for every column so numeric-looking
        # object columns can still get min, max, mean, and infinity checks.
        numeric = pd.to_numeric(series, errors="coerce")
        infinite_mask = np.isinf(numeric)
        stats["infinite_count"] += int(infinite_mask.sum())

        # Infinite values are counted above but excluded from numeric summaries.
        clean_numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna()
        if clean_numeric.empty:
            continue

        stats["numeric_count"] += int(clean_numeric.count())
        stats["numeric_sum"] += float(clean_numeric.sum())

        chunk_min = float(clean_numeric.min())
        chunk_max = float(clean_numeric.max())
        stats["min"] = chunk_min if pd.isna(stats["min"]) else min(float(stats["min"]), chunk_min)
        stats["max"] = chunk_max if pd.isna(stats["max"]) else max(float(stats["max"]), chunk_max)


def build_summary_rows(
    rows_seen: int,
    file_count: int,
    failed_file_count: int,
    feature_stats: dict[str, dict[str, object]],
    label_counts: Counter[str],
    label_column: str | None,
) -> list[dict[str, object]]:
    """Build one combined CSV-friendly summary for dataset, features, labels."""
    rows: list[dict[str, object]] = [
        {
            "section": "dataset",
            "name": "total_rows",
            "count": rows_seen,
            "total_rows": rows_seen,
            "non_null_count": "",
            "missing_count": "",
            "infinite_count": "",
            "dtype": "",
            "min": "",
            "max": "",
            "mean": "",
            "label_column": label_column or "",
        },
        {
            "section": "dataset",
            "name": "files",
            "count": file_count,
            "total_rows": rows_seen,
            "non_null_count": "",
            "missing_count": "",
            "infinite_count": "",
            "dtype": "",
            "min": "",
            "max": "",
            "mean": "",
            "label_column": label_column or "",
        },
        {
            "section": "dataset",
            "name": "failed_files",
            "count": failed_file_count,
            "total_rows": rows_seen,
            "non_null_count": "",
            "missing_count": "",
            "infinite_count": "",
            "dtype": "",
            "min": "",
            "max": "",
            "mean": "",
            "label_column": label_column or "",
        },
        {
            "section": "dataset",
            "name": "feature_count",
            "count": len(feature_stats),
            "total_rows": rows_seen,
            "non_null_count": "",
            "missing_count": "",
            "infinite_count": "",
            "dtype": "",
            "min": "",
            "max": "",
            "mean": "",
            "label_column": label_column or "",
        },
        {
            "section": "dataset",
            "name": "label_class_count",
            "count": len(label_counts),
            "total_rows": rows_seen,
            "non_null_count": "",
            "missing_count": "",
            "infinite_count": "",
            "dtype": "",
            "min": "",
            "max": "",
            "mean": "",
            "label_column": label_column or "",
        },
    ]

    for feature_name, stats in sorted(feature_stats.items()):
        numeric_count = int(stats["numeric_count"])
        rows.append(
            {
                "section": "feature",
                "name": feature_name,
                "count": int(stats["non_null_count"]),
                "total_rows": rows_seen,
                "non_null_count": int(stats["non_null_count"]),
                "missing_count": int(stats["missing_count"]),
                "infinite_count": int(stats["infinite_count"]),
                "dtype": stats["dtype"],
                "min": stats["min"] if numeric_count else "",
                "max": stats["max"] if numeric_count else "",
                "mean": float(stats["numeric_sum"]) / numeric_count if numeric_count else "",
                "label_column": label_column or "",
            }
        )

    for label, count in sorted(label_counts.items(), key=lambda item: item[1], reverse=True):
        rows.append(
            {
                "section": "label",
                "name": label,
                "count": count,
                "total_rows": rows_seen,
                "non_null_count": "",
                "missing_count": "",
                "infinite_count": "",
                "dtype": "",
                "min": "",
                "max": "",
                "mean": "",
                "label_column": label_column or "",
            }
        )

    return rows


def inspect_dataset_group(
    dataset_name: str,
    dataset_files: list[Path],
    chunksize: int,
    label_column: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, str]]]:
    """Inspect one dataset group and return summary and label dataframes."""
    label_counts: Counter[str] = Counter()
    feature_stats: dict[str, dict[str, object]] = {}
    file_errors: list[dict[str, str]] = []
    rows_seen = 0
    detected_label_column: str | None = None

    print(f"\nDataset: {dataset_name}")

    for dataset_file in dataset_files:
        print(f"Reading {format_path_for_display(dataset_file)}")

        try:
            for chunk in read_dataset_file(dataset_file, chunksize=chunksize):
                chunk = clean_columns(chunk)
                update_feature_stats(feature_stats, chunk)

                current_label_column = detect_label_column(list(chunk.columns), label_column)
                if current_label_column:
                    detected_label_column = detected_label_column or current_label_column
                    label_counts.update(clean_label_values(chunk[current_label_column]))

                rows_seen += len(chunk)
        except Exception as error:
            file_errors.append(
                {
                    "file": format_path_for_display(dataset_file),
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
            print(f"Skipping unreadable file: {format_path_for_display(dataset_file)} ({type(error).__name__})")

    label_counts_df = pd.DataFrame(
        sorted(label_counts.items(), key=lambda item: item[1], reverse=True),
        columns=["label", "count"],
    )
    summary_rows = build_summary_rows(
        rows_seen=rows_seen,
        file_count=len(dataset_files),
        failed_file_count=len(file_errors),
        feature_stats=feature_stats,
        label_counts=label_counts,
        label_column=detected_label_column,
    )
    summary_df = pd.DataFrame(summary_rows)

    print(f"\nRows read: {rows_seen:,}")
    print(f"Files read: {len(dataset_files)}")
    print(f"Files skipped after read errors: {len(file_errors)}")
    print(f"Features found: {len(feature_stats)}")
    print(f"Label column: {detected_label_column or 'not found'}")
    print(f"Label classes found: {len(label_counts)}")

    if not label_counts_df.empty:
        print("\nTop labels:")
        print(label_counts_df.head(10).to_string(index=False))

    return summary_df, label_counts_df, file_errors


def write_excel_report(
    report_path: Path,
    dataset_reports: dict[str, pd.DataFrame],
    all_errors: list[dict[str, str]],
) -> None:
    """Write one workbook with one tab per dataset plus an errors tab."""
    used_sheet_names: set[str] = set()
    with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
        for dataset_name, summary_df in dataset_reports.items():
            sheet_name = safe_sheet_name(dataset_name, used_sheet_names)
            summary_df.to_excel(writer, sheet_name=sheet_name, index=False)

        if all_errors:
            errors_df = pd.DataFrame(all_errors, columns=["dataset", "file", "error_type", "error_message"])
        else:
            errors_df = pd.DataFrame(columns=["dataset", "file", "error_type", "error_message"])
        errors_df.to_excel(writer, sheet_name=safe_sheet_name("file_errors", used_sheet_names), index=False)


def inspect_datasets(
    input_path: Path,
    output_dir: Path,
    chunksize: int,
    label_column: str | None,
    output_prefix: str,
    exclude_patterns: list[str],
) -> None:
    """Scan raw datasets and write a multi-tab Excel report plus CSV copies."""
    dataset_files = find_dataset_files(input_path, exclude_patterns)
    if not dataset_files:
        raise FileNotFoundError(f"No supported dataset files found in {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    grouped_files = group_dataset_files(input_path, dataset_files)
    dataset_reports: dict[str, pd.DataFrame] = {}
    all_errors: list[dict[str, str]] = []

    print(f"Inspecting dataset files from: {input_path}")
    if exclude_patterns:
        print(f"Excluding paths containing: {', '.join(exclude_patterns)}")

    for dataset_name, files in sorted(grouped_files.items()):
        summary_df, label_counts_df, file_errors = inspect_dataset_group(
            dataset_name=dataset_name,
            dataset_files=files,
            chunksize=chunksize,
            label_column=label_column,
        )
        dataset_reports[dataset_name] = summary_df

        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", dataset_name).strip("_") or "dataset"
        summary_df.to_csv(output_dir / f"{output_prefix}_{safe_name}_labels_features_counts.csv", index=False)
        label_counts_df.to_csv(output_dir / f"{output_prefix}_{safe_name}_label_counts.csv", index=False)

        for error in file_errors:
            all_errors.append({"dataset": dataset_name, **error})

    report_path = output_dir / f"{output_prefix}_dataset_understanding_report.xlsx"
    write_excel_report(report_path, dataset_reports, all_errors)
    pd.DataFrame(all_errors, columns=["dataset", "file", "error_type", "error_message"]).to_csv(
        output_dir / f"{output_prefix}_file_errors.csv",
        index=False,
    )

    print(f"\nExcel report written to: {report_path}")


def parse_args() -> argparse.Namespace:
    """Read command-line options for the generic inspection script."""
    parser = argparse.ArgumentParser(
        description="Inspect tabular datasets for rows, features, labels, missing values, and numeric summaries."
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument(
        "--label-column",
        default=None,
        help="Optional label/target column. If omitted, common names such as Label, label, target, and class are detected.",
    )
    parser.add_argument("--output-prefix", default="public_datasets")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Skip files whose path contains this text. Can be used more than once.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    inspect_datasets(
        input_path=args.input_path,
        output_dir=args.output_dir,
        chunksize=args.chunksize,
        label_column=args.label_column,
        output_prefix=args.output_prefix,
        exclude_patterns=args.exclude,
    )
