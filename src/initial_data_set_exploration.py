"""first look at the raw public datasets."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "Zenodo_13754300"
FIRST_ROWS_TO_SHOW = 5


csv_files = sorted(RAW_DATA_DIR.rglob("*.csv"))

print(f"Raw data folder: {RAW_DATA_DIR}")
print(f"CSV files found: {len(csv_files)}")

for csv_file in csv_files:
    print("\n" + "=" * 80)
    print(f"File: {csv_file}")

    dataset = pd.read_csv(csv_file)
    dataset.columns = dataset.columns.str.strip()

    print("\nData shape:")
    print(f"Rows: {dataset.shape[0]}")
    print(f"Columns: {dataset.shape[1]}")

    print("\nColumn names:")
    print(list(dataset.columns))

    print("\nData types:")
    print(dataset.dtypes)

    print(f"\nFirst {FIRST_ROWS_TO_SHOW} rows:")
    print(dataset.head(FIRST_ROWS_TO_SHOW))
