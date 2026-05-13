"""Quick first look at the raw Zenodo public dataset files.

This script is intentionally simple: it does not clean, transform, or save any
data. It only prints the basics needed before writing processing code.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "Zenodo_13754300"
FIRST_ROWS_TO_SHOW = 5


# Recursively find every CSV under the raw Zenodo folder so newly added files
# are included automatically.
csv_files = sorted(RAW_DATA_DIR.rglob("*.csv"))

print(f"Raw data folder: {RAW_DATA_DIR}")
print(f"CSV files found: {len(csv_files)}")

for csv_file in csv_files:
    print("\n" + "=" * 80)
    print(f"File: {csv_file}")

    # Read one raw file as-is. Column stripping avoids confusing duplicates
    # caused only by accidental whitespace in CSV headers.
    dataset = pd.read_csv(csv_file)
    dataset.columns = dataset.columns.str.strip()

    # Shape tells us the number of observations and fields in the raw file.
    print("\nData shape:")
    print(f"Rows: {dataset.shape[0]}")
    print(f"Columns: {dataset.shape[1]}")

    # Column names and dtypes are the main inputs for the later processing
    # scripts, because they tell us what can become features or targets.
    print("\nColumn names:")
    print(list(dataset.columns))

    print("\nData types:")
    print(dataset.dtypes)

    # The first rows are only for human inspection, not modelling.
    print(f"\nFirst {FIRST_ROWS_TO_SHOW} rows:")
    print(dataset.head(FIRST_ROWS_TO_SHOW))
