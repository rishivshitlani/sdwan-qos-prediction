"""Train baseline regression models for SD-WAN QoS prediction.

This script is the first modelling step for the project. It loads a project-style
CSV dataset, trains baseline and tree/boosting models, then writes comparable
holdout-test and k-fold cross-validation metrics to
``reports/model_results/model_results.csv``.
"""

from __future__ import annotations

import argparse
import warnings
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
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


def build_preprocessor(features: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    """Build preprocessing for numeric and categorical project features."""
    # Linear models benefit from scaled numeric inputs. Tree-based models do not
    # need scaling, so their preprocessor only imputes numeric values.
    numeric_columns = features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [column for column in features.columns if column not in numeric_columns]

    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        # Scaling is enabled only for models such as Linear Regression where
        # feature magnitude affects the fitted coefficients.
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)
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


def build_model_specs(random_state: int) -> list[tuple[str, object | None, str, bool]]:
    """Return models to evaluate, including a graceful XGBoost fallback."""
    # The final boolean tells the trainer whether the model should receive
    # scaled numeric features. This keeps preprocessing tailored to the model
    # family instead of applying StandardScaler to every estimator.
    models: list[tuple[str, object | None, str, bool]] = [
        # DummyRegressor is the sanity-check baseline: real models should beat
        # this simple "always predict the training mean" strategy.
        ("DummyRegressor_mean", DummyRegressor(strategy="mean"), "", False),
        ("LinearRegression", LinearRegression(), "", True),
        (
            "RandomForestRegressor",
            RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1),
            "",
            False,
        ),
    ]

    # XGBoost is a required advanced baseline for the project, but its Python
    # package depends on a native OpenMP library on macOS. If that runtime is
    # missing locally, keep the training run usable and record the skip reason
    # in the results CSV.
    try:
        from xgboost import XGBRegressor
    except Exception as error:  # pragma: no cover - depends on local native libs
        models.append(("XGBRegressor", None, format_xgboost_error(error), False))
    else:
        models.append(
            (
                "XGBRegressor",
                XGBRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=4,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="reg:squarederror",
                    eval_metric="rmse",
                    random_state=random_state,
                    n_jobs=-1,
                    verbosity=0,
                ),
                "",
                False,
            )
        )

    return models


def format_xgboost_error(error: Exception) -> str:
    """Make XGBoost import/runtime errors understandable in the results CSV."""
    error_type = type(error).__name__
    error_text = str(error)

    if isinstance(error, ModuleNotFoundError):
        return (
            f"{error_type}: xgboost is not installed for this Python interpreter. "
            "Run with the project virtual environment, for example `.venv/bin/python src/train_zenodo_baseline.py`."
        )

    if "libomp" in error_text or "OpenMP" in error_text:
        return (
            f"{error_type}: native XGBoost library could not load because OpenMP/libomp is missing or not linked. "
            "On macOS, run `brew install libomp` and `brew link --force libomp`."
        )

    return f"{error_type}: {error_text}"


def model_status_result(
    model_name: str,
    status: str,
    message: str,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    dataset_name: str,
    target_column: str,
    cv_folds: int,
) -> dict[str, object]:
    """Create an output row for a model that cannot run in this environment."""
    return {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": dataset_name,
        "model": model_name,
        "target": target_column,
        "status": status,
        "error_message": message,
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "feature_count": x_train.shape[1],
        "mae": None,
        "rmse": None,
        "r2_score": None,
        "cv_folds": cv_folds,
        "cv_mae_mean": None,
        "cv_mae_std": None,
        "cv_rmse_mean": None,
        "cv_rmse_std": None,
        "cv_r2_mean": None,
        "cv_r2_std": None,
        "training_time_sec": None,
        "inference_time_sec": None,
        "inference_time_ms_per_row": None,
    }


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
    cv_folds: int,
    random_state: int,
) -> dict[str, object]:
    """Train one model and return regression metrics for the selected target."""
    # The pipeline ensures preprocessing learned on the training split is reused
    # consistently on the test split. A fresh preprocessor clone is passed into
    # this function for each model, so fitted transformer state is never shared
    # between estimators.
    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    # Cross-validation is run on the training split only. This gives a more
    # stable estimate than one train/test split while keeping the final holdout
    # test set untouched for the reported test metrics below.
    cv_results: dict[str, object] = {
        "cv_folds": cv_folds,
        "cv_mae_mean": None,
        "cv_mae_std": None,
        "cv_rmse_mean": None,
        "cv_rmse_std": None,
        "cv_r2_mean": None,
        "cv_r2_std": None,
    }
    if cv_folds > 1:
        if cv_folds > len(x_train):
            raise ValueError(
                f"cv_folds ({cv_folds}) cannot be greater than the number of training rows ({len(x_train)})."
            )

        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

        # cross_val_score clones the whole pipeline for each fold. This means
        # imputation, encoding, and optional scaling are fitted only on the
        # training portion of that fold, which avoids validation-fold leakage.
        #
        # scikit-learn returns negative values for error metrics because higher
        # scores are considered better. Convert them back to positive errors for
        # easier interpretation in the CSV output.
        # Some Python/numpy/scikit-learn combinations emit RuntimeWarnings for
        # numerically unstable LinearRegression matrix multiplication even when
        # the final metrics are finite. Suppress those warnings here and record
        # actual failures through exceptions/results instead of noisy console
        # output.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            cv_mae = -cross_val_score(pipeline, x_train, y_train, cv=cv, scoring="neg_mean_absolute_error")
            cv_rmse = -cross_val_score(
                pipeline,
                x_train,
                y_train,
                cv=cv,
                scoring="neg_root_mean_squared_error",
            )
            cv_r2 = cross_val_score(pipeline, x_train, y_train, cv=cv, scoring="r2")

        cv_results.update(
            {
                "cv_mae_mean": float(cv_mae.mean()),
                "cv_mae_std": float(cv_mae.std()),
                "cv_rmse_mean": float(cv_rmse.mean()),
                "cv_rmse_std": float(cv_rmse.std()),
                "cv_r2_mean": float(cv_r2.mean()),
                "cv_r2_std": float(cv_r2.std()),
            }
        )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        train_start = perf_counter()
        pipeline.fit(x_train, y_train)
        training_time_sec = perf_counter() - train_start

        inference_start = perf_counter()
        predictions = pipeline.predict(x_test)
        inference_time_sec = perf_counter() - inference_start

    if not np.isfinite(predictions).all():
        raise ValueError("Model produced non-finite predictions.")

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
        "status": "ok",
        "error_message": "",
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "feature_count": x_train.shape[1],
        "mae": mae,
        "rmse": rmse,
        "r2_score": r2,
        **cv_results,
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
    cv_folds: int,
) -> pd.DataFrame:
    """Train baseline models, then save holdout and cross-validation metrics."""
    data = load_dataset(input_path, target_column)
    features, target = split_features_and_target(data, target_column, drop_columns)

    # Use a fixed random_state so baseline results are repeatable in reports.
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )

    # Build two unfitted preprocessing templates:
    # - scaled: for Linear Regression
    # - unscaled: for tree/boosting models where scaling is unnecessary
    scaled_preprocessor = build_preprocessor(features, scale_numeric=True)
    unscaled_preprocessor = build_preprocessor(features, scale_numeric=False)
    results = []
    for model_name, model, skip_reason, scale_numeric in build_model_specs(random_state):
        if model is None:
            results.append(
                model_status_result(
                    model_name=model_name,
                    status="skipped",
                    message=skip_reason,
                    x_train=x_train,
                    x_test=x_test,
                    dataset_name=dataset_name,
                    target_column=target_column,
                    cv_folds=cv_folds,
                )
            )
            continue

        try:
            # clone(...) is important because Pipeline.fit mutates the
            # preprocessor in place. Without cloning, later models could reuse
            # fitted state from earlier model runs.
            results.append(
                evaluate_model(
                    model_name=model_name,
                    model=model,
                    # Each model receives a fresh clone so no fitted state is
                    # shared across model runs.
                    preprocessor=clone(scaled_preprocessor if scale_numeric else unscaled_preprocessor),
                    x_train=x_train,
                    x_test=x_test,
                    y_train=y_train,
                    y_test=y_test,
                    dataset_name=dataset_name,
                    target_column=target_column,
                    cv_folds=cv_folds,
                    random_state=random_state,
                )
            )
        except Exception as error:
            results.append(
                model_status_result(
                    model_name=model_name,
                    status="failed",
                    message=f"{type(error).__name__}: {error}",
                    x_train=x_train,
                    x_test=x_test,
                    dataset_name=dataset_name,
                    target_column=target_column,
                    cv_folds=cv_folds,
                )
            )

    results_df = pd.DataFrame(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    return results_df


def parse_args() -> argparse.Namespace:
    """Read command-line options for baseline training."""
    parser = argparse.ArgumentParser(
        description="Train baseline regression models for QoS prediction."
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--target-column", default=TARGET_COLUMN)
    parser.add_argument("--dataset-name", default="synthetic_sdwan")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of k-fold cross-validation folds to run on the training split. Use 0 or 1 to disable.",
    )
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
        cv_folds=args.cv_folds,
    )

    print(f"Model results written to: {args.output_path}")
    print(metric_results.to_string(index=False))
