"""Train baseline regression models for SD-WAN QoS prediction.

Loads a project-style CSV, trains DummyRegressor, Linear Regression, SVR,
Random Forest, and XGBoost, then appends holdout and k-fold cross-validation
metrics plus feature-importance output to reports/model_results/.
"""

from __future__ import annotations

import argparse
import warnings
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import NamedTuple

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
from sklearn.svm import SVR


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "synthetic" / "sdwan_qos_synthetic.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "reports" / "model_results" / "model_results.csv"
TARGET_COLUMN = "recommended_bandwidth_percent"
DEFAULT_DROP_COLUMNS = [
    "cicids_attack_label",
    "source_file",
    "destination_port",
    "byte_count",
]


class ModelSpec(NamedTuple):
    name: str
    model: object | None
    scale_numeric: bool
    skip_reason: str = ""


def build_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_dataset(input_path: Path, target_column: str) -> pd.DataFrame:
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
    columns_to_drop = [c for c in drop_columns if c in data.columns]
    features = data.drop(columns=[target_column, *columns_to_drop])
    target = pd.to_numeric(data[target_column], errors="coerce")
    mask = target.notna()
    return features.loc[mask].copy(), target.loc[mask].copy()


def build_preprocessor(features: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in features.columns if c not in numeric_cols]

    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(steps=numeric_steps), numeric_cols),
            (
                "categorical",
                Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", build_one_hot_encoder()),
                ]),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )


def build_model_specs(random_state: int) -> list[ModelSpec]:
    specs = [
        ModelSpec("DummyRegressor_mean", DummyRegressor(strategy="mean"), scale_numeric=False),
        ModelSpec("LinearRegression", LinearRegression(), scale_numeric=True),
        ModelSpec("SVR_rbf", SVR(kernel="rbf", C=10.0, epsilon=0.1), scale_numeric=True),
        ModelSpec(
            "RandomForestRegressor",
            RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1),
            scale_numeric=False,
        ),
    ]

    try:
        from xgboost import XGBRegressor
    except Exception as exc:
        specs.append(ModelSpec("XGBRegressor", None, scale_numeric=False, skip_reason=_xgb_error(exc)))
    else:
        specs.append(ModelSpec(
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
            scale_numeric=False,
        ))

    return specs


def _xgb_error(exc: Exception) -> str:
    name, text = type(exc).__name__, str(exc)
    if isinstance(exc, ModuleNotFoundError):
        return (
            f"{name}: xgboost is not installed. "
            "Run with the project virtual environment: `.venv/bin/python src/train_zenodo_baseline.py`."
        )
    if "libomp" in text or "OpenMP" in text:
        return (
            f"{name}: OpenMP/libomp is missing or not linked. "
            "On macOS run `brew install libomp && brew link --force libomp`."
        )
    return f"{name}: {text}"


def _skipped_result(
    spec: ModelSpec,
    status: str,
    message: str,
    train_rows: int,
    test_rows: int,
    feature_count: int,
    dataset_name: str,
    target_column: str,
    cv_folds: int,
) -> dict[str, object]:
    return {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": dataset_name,
        "model": spec.name,
        "target": target_column,
        "status": status,
        "error_message": message,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "feature_count": feature_count,
        "mae": None, "rmse": None, "r2_score": None,
        "cv_folds": cv_folds,
        "cv_mae_mean": None, "cv_mae_std": None,
        "cv_rmse_mean": None, "cv_rmse_std": None,
        "cv_r2_mean": None, "cv_r2_std": None,
        "training_time_sec": None,
        "inference_time_sec": None,
        "inference_time_ms_per_row": None,
    }


def _run_cv(
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int,
    random_state: int,
) -> dict[str, object]:
    if cv_folds <= 1:
        return {"cv_folds": cv_folds, "cv_mae_mean": None, "cv_mae_std": None,
                "cv_rmse_mean": None, "cv_rmse_std": None, "cv_r2_mean": None, "cv_r2_std": None}

    if cv_folds > len(x_train):
        raise ValueError(f"cv_folds ({cv_folds}) exceeds training rows ({len(x_train)}).")

    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    # cross_val_score clones the pipeline per fold, so imputation and scaling are
    # fitted only on each fold's training portion — no validation-fold leakage.
    # Negative-metric convention: scikit-learn returns negated error scores so
    # that higher is always better; negate again to get positive MAE/RMSE.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        cv_mae  = -cross_val_score(pipeline, x_train, y_train, cv=cv, scoring="neg_mean_absolute_error")
        cv_rmse = -cross_val_score(pipeline, x_train, y_train, cv=cv, scoring="neg_root_mean_squared_error")
        cv_r2   =  cross_val_score(pipeline, x_train, y_train, cv=cv, scoring="r2")

    return {
        "cv_folds": cv_folds,
        "cv_mae_mean":  float(cv_mae.mean()),  "cv_mae_std":  float(cv_mae.std()),
        "cv_rmse_mean": float(cv_rmse.mean()), "cv_rmse_std": float(cv_rmse.std()),
        "cv_r2_mean":   float(cv_r2.mean()),   "cv_r2_std":   float(cv_r2.std()),
    }


def _feature_importance_rows(
    pipeline: Pipeline,
    spec: ModelSpec,
    dataset_name: str,
    target_column: str,
) -> list[dict[str, object]]:
    """Extract feature importances from fitted RF/XGBoost pipelines."""
    model = pipeline.named_steps["model"]
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return []

    preprocessor = pipeline.named_steps["preprocess"]
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    total = float(np.sum(importances))
    normalized = importances / total if total > 0 else importances

    rows = []
    for rank, (feature, importance, normalized_importance) in enumerate(
        sorted(
            zip(feature_names, importances, normalized),
            key=lambda item: item[1],
            reverse=True,
        ),
        start=1,
    ):
        rows.append(
            {
                "run_timestamp": datetime.now().isoformat(timespec="seconds"),
                "dataset": dataset_name,
                "model": spec.name,
                "target": target_column,
                "rank": rank,
                "feature": feature,
                "importance": float(importance),
                "importance_normalized": float(normalized_importance),
            }
        )
    return rows


def evaluate_model(
    spec: ModelSpec,
    preprocessor: ColumnTransformer,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    dataset_name: str,
    target_column: str,
    cv_folds: int,
    random_state: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", spec.model)])

    cv_results = _run_cv(pipeline, x_train, y_train, cv_folds, random_state)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        t0 = perf_counter()
        pipeline.fit(x_train, y_train)
        training_time_sec = perf_counter() - t0

        t1 = perf_counter()
        predictions = pipeline.predict(x_test)
        inference_time_sec = perf_counter() - t1

    if not np.isfinite(predictions).all():
        raise ValueError("Model produced non-finite predictions.")

    result = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": dataset_name,
        "model": spec.name,
        "target": target_column,
        "status": "ok",
        "error_message": "",
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "feature_count": x_train.shape[1],
        "mae":      mean_absolute_error(y_test, predictions),
        "rmse":     float(np.sqrt(mean_squared_error(y_test, predictions))),
        "r2_score": r2_score(y_test, predictions),
        **cv_results,
        "training_time_sec":       training_time_sec,
        "inference_time_sec":      inference_time_sec,
        "inference_time_ms_per_row": inference_time_sec / len(x_test) * 1000,
    }
    return result, _feature_importance_rows(pipeline, spec, dataset_name, target_column)


def feature_importance_output_path(output_path: Path) -> Path:
    """Create the sibling CSV path for model feature importances."""
    return output_path.with_name(f"{output_path.stem}_feature_importance{output_path.suffix}")


def append_results_csv(rows: pd.DataFrame, output_path: Path) -> None:
    """Append result rows to a CSV, writing the header only for a new file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    append_to_existing = output_path.exists() and output_path.stat().st_size > 0
    rows.to_csv(
        output_path,
        mode="a" if append_to_existing else "w",
        header=not append_to_existing,
        index=False,
    )


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
    data = load_dataset(input_path, target_column)
    features, target = split_features_and_target(data, target_column, drop_columns)

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state,
    )

    preprocessors = {
        True:  build_preprocessor(features, scale_numeric=True),
        False: build_preprocessor(features, scale_numeric=False),
    }

    results = []
    feature_importance_rows: list[dict[str, object]] = []
    for spec in build_model_specs(random_state):
        skipped_kwargs = dict(
            spec=spec,
            train_rows=len(x_train),
            test_rows=len(x_test),
            feature_count=x_train.shape[1],
            dataset_name=dataset_name,
            target_column=target_column,
            cv_folds=cv_folds,
        )

        if spec.model is None:
            results.append(_skipped_result(status="skipped", message=spec.skip_reason, **skipped_kwargs))
            continue

        try:
            result, model_feature_importances = evaluate_model(
                spec=spec,
                preprocessor=clone(preprocessors[spec.scale_numeric]),
                x_train=x_train, x_test=x_test,
                y_train=y_train, y_test=y_test,
                dataset_name=dataset_name,
                target_column=target_column,
                cv_folds=cv_folds,
                random_state=random_state,
            )
            results.append(result)
            feature_importance_rows.extend(model_feature_importances)
        except Exception as exc:
            results.append(
                _skipped_result(
                    status="failed",
                    message=f"{type(exc).__name__}: {exc}",
                    **skipped_kwargs,
                )
            )

    results_df = pd.DataFrame(results)
    append_results_csv(results_df, output_path)
    if feature_importance_rows:
        append_results_csv(pd.DataFrame(feature_importance_rows), feature_importance_output_path(output_path))
    return results_df


def parse_args() -> argparse.Namespace:
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
        "--cv-folds", type=int, default=5,
        help="K-fold CV folds on the training split. Use 0 or 1 to disable.",
    )
    parser.add_argument(
        "--drop-column", action="append", default=[],
        help="Input column to exclude from training. Repeatable.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = train_baselines(
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
    print(results.to_string(index=False))
