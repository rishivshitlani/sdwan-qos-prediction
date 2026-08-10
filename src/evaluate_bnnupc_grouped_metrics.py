"""Grouped cross-validation for BNN-UPC tail-delay and jitter targets."""

from __future__ import annotations

import argparse
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from evaluate_bnnupc_metric_slices import (
    DEFAULT_INPUT_PATH,
    split_features_and_target,
)
from evaluate_bnnupc_qos_slices import model_specs_by_name
from train_baseline import build_preprocessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "reports" / "model_results" / "bnnupc_grouped_cv_metrics.csv"
)
DEFAULT_TARGETS = ["delay_p90", "jitter"]
MODEL_NAME = "XGBRegressor"
QOS_CLASSES = ["Gold", "Silver", "Bronze"]


def grouped_predictions(
    features: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series,
    *,
    cv_folds: int,
    random_state: int,
) -> np.ndarray:
    spec = model_specs_by_name(random_state)[MODEL_NAME]
    preprocessor = build_preprocessor(features, scale_numeric=spec.scale_numeric)
    predictions = np.empty(len(features), dtype=float)

    for train_idx, test_idx in GroupKFold(n_splits=cv_folds).split(
        features, target, groups
    ):
        pipeline = Pipeline(
            [
                ("preprocess", clone(preprocessor)),
                ("model", clone(spec.model)),
            ]
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            pipeline.fit(features.iloc[train_idx], target.iloc[train_idx])
        predictions[test_idx] = pipeline.predict(features.iloc[test_idx])
    return predictions


def evaluate_targets(
    input_path: Path,
    output_path: Path,
    *,
    targets: list[str],
    cv_folds: int,
    random_state: int,
) -> pd.DataFrame:
    data = pd.read_csv(input_path)
    if "simulation_id" not in data.columns:
        raise ValueError("Grouped evaluation requires simulation_id.")

    rows: list[dict[str, object]] = []
    timestamp = datetime.now().isoformat(timespec="seconds")
    for target_name in targets:
        features, target = split_features_and_target(data, target_name)
        groups = data.loc[target.index, "simulation_id"]
        predictions = grouped_predictions(
            features,
            target,
            groups,
            cv_folds=cv_folds,
            random_state=random_state,
        )

        labels = data.loc[target.index, "qos_class"].to_numpy()
        for qos_class in ["Overall", *QOS_CLASSES]:
            mask = np.ones(len(target), dtype=bool) if qos_class == "Overall" else labels == qos_class
            truth = target.to_numpy()[mask]
            estimate = predictions[mask]
            rows.append(
                {
                    "run_timestamp": timestamp,
                    "evaluation": "GroupKFold_by_simulation_id",
                    "model": MODEL_NAME,
                    "target": target_name,
                    "qos_class": qos_class,
                    "rows": int(mask.sum()),
                    "mae_ms": float(mean_absolute_error(truth, estimate) * 1000.0),
                    "r2": float(r2_score(truth, estimate)),
                    "cv_folds": cv_folds,
                }
            )

    result = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate tail delay and jitter with simulation-grouped CV."
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--target", action="append", default=[])
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = evaluate_targets(
        args.input_path,
        args.output_path,
        targets=args.target or DEFAULT_TARGETS,
        cv_folds=args.cv_folds,
        random_state=args.random_state,
    )
    print(results.to_string(index=False))
