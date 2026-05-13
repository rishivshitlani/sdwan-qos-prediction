"""Train first baseline regression models for SD-WAN QoS prediction.

This script is the first modelling step for the project. It loads a project-style
CSV dataset, trains a simple DummyRegressor and Linear Regression model, then
writes comparable metrics to ``reports/model_results/model_results.csv``.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "synthetic" / "sdwan_qos_synthetic.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "reports" / "model_results" / "model_results.csv"
TARGET_COLUMN = "recommended_bandwidth_percent"
DEFAULT_DROP_COLUMNS = [
    # These CICIDS traceability columns are useful for auditing rows, but they
    # should not be used as model inputs for QoS bandwidth prediction.
    "cicids_attack_label",
    "source_file",
    "destination_port",
    "byte_count",
]


def build_one_hot_encoder() -> OneHotEncoder:
    """Create a OneHotEncoder compatible with different scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_dataset(input_path: Path, target_column: str) -> pd.DataFrame:
    """Read the modelling dataset and confirm the target column exists."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset does not exist: {input_path}")

    data = pd.read_csv(input_path)
    data.columns = data.columns.astype(str).str.strip()

    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in {input_path}")

    return data


def split_features_and_target(
    data: pd.DataFrame,
    target_column: str,
    drop_columns: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model inputs from the regression target."""
    # Only drop columns that are present so the same trainer can be reused with
    # synthetic, CICIDS-derived, and Zenodo datasets.
    columns_to_drop = [column for column in drop_columns if column in data.columns]
    features = data.drop(columns=[target_column, *columns_to_drop])

    # Regression metrics require a numeric target. Non-numeric or missing target
    # values are converted to NaN and removed below.
    target = pd.to_numeric(data[target_column], errors="coerce")

    valid_target_mask = target.notna()
    return features.loc[valid_target_mask].copy(), target.loc[valid_target_mask].copy()


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing for numeric and categorical project features."""
    # Numeric columns can be scaled directly; object/string columns need one-hot
    # encoding so Linear Regression can use them.
    numeric_columns = features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [column for column in features.columns if column not in numeric_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", build_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        # Drop anything not explicitly routed through the numeric or categorical
        # pipelines. This keeps the feature matrix predictable.
        remainder="drop",
    )


def evaluate_model(
    model_name: str,
    model: object,
    preprocessor: ColumnTransformer,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    dataset_name: str,
    target_column: str,
) -> dict[str, object]:
    """Train one model and return regression metrics for the selected target."""
    # The pipeline ensures preprocessing learned on the training split is reused
    # consistently on the test split.
    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    train_start = perf_counter()
    pipeline.fit(x_train, y_train)
    training_time_sec = perf_counter() - train_start

    inference_start = perf_counter()
    predictions = pipeline.predict(x_test)
    inference_time_sec = perf_counter() - inference_start

    # MAE/RMSE show error in target units. R2 shows how much variance the model
    # explains compared with predicting the mean.
    mae = mean_absolute_error(y_test, predictions)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    r2 = r2_score(y_test, predictions)

    return {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": dataset_name,
        "model": model_name,
        "target": target_column,
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "feature_count": x_train.shape[1],
        "mae": mae,
        "rmse": rmse,
        "r2_score": r2,
        "training_time_sec": training_time_sec,
        "inference_time_sec": inference_time_sec,
        "inference_time_ms_per_row": inference_time_sec / len(x_test) * 1000,
    }


def train_baselines(
    input_path: Path,
    output_path: Path,
    target_column: str,
    dataset_name: str,
    test_size: float,
    random_state: int,
    drop_columns: list[str],
) -> pd.DataFrame:
    """Train DummyRegressor and Linear Regression, then save metric results."""
    data = load_dataset(input_path, target_column)
    features, target = split_features_and_target(data, target_column, drop_columns)

    # Use a fixed random_state so baseline results are repeatable in reports.
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )

    preprocessor = build_preprocessor(features)
    models = [
        # DummyRegressor is the sanity-check baseline: real models should beat
        # this simple "always predict the training mean" strategy.
        ("DummyRegressor_mean", DummyRegressor(strategy="mean")),
        ("LinearRegression", LinearRegression()),
    ]

    results = [
        evaluate_model(
            model_name=model_name,
            model=model,
            preprocessor=preprocessor,
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
            dataset_name=dataset_name,
            target_column=target_column,
        )
        for model_name, model in models
    ]

    results_df = pd.DataFrame(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    return results_df


def parse_args() -> argparse.Namespace:
    """Read command-line options for baseline training."""
    parser = argparse.ArgumentParser(
        description="Train DummyRegressor and Linear Regression baselines for QoS bandwidth prediction."
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--target-column", default=TARGET_COLUMN)
    parser.add_argument("--dataset-name", default="synthetic_sdwan")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--drop-column",
        action="append",
        default=[],
        help="Extra input column to exclude from model training. Can be used more than once.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    metric_results = train_baselines(
        input_path=args.input_path,
        output_path=args.output_path,
        target_column=args.target_column,
        dataset_name=args.dataset_name,
        test_size=args.test_size,
        random_state=args.random_state,
        drop_columns=DEFAULT_DROP_COLUMNS + args.drop_column,
    )

    print(f"Model results written to: {args.output_path}")
    print(metric_results.to_string(index=False))
