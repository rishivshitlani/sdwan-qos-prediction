"""Simulation-grouped cross-validation for the BNN-UPC delay model.

Each BNNetSimulator run produces multiple flow rows.  This evaluator keeps all
rows from a simulation_id in the same fold so that performance reflects
generalisation to unseen simulation runs rather than unseen rows from familiar
runs.  It reports overall/per-class regression metrics and per-class SLA
violation metrics for XGBoost.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from evaluate_bnnupc_qos_slices import (
    DEFAULT_INPUT_PATH,
    DEFAULT_SLA_MS,
    build_sla_rows,
    build_slice_rows,
    delay_ms_from_log,
    load_bnnupc_dataset,
    model_specs_by_name,
    split_features_and_target,
)
from train_baseline import build_preprocessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "reports" / "model_results" / "bnnupc_grouped_cv_xgb.csv"
)
MODEL_NAME = "XGBRegressor"


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
    cv = GroupKFold(n_splits=cv_folds)

    for train_idx, test_idx in cv.split(features, target, groups):
        pipeline = Pipeline(
            steps=[
                ("preprocess", clone(preprocessor)),
                ("model", clone(spec.model)),
            ]
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            pipeline.fit(features.iloc[train_idx], target.iloc[train_idx])
        predictions[test_idx] = pipeline.predict(features.iloc[test_idx])

    return predictions


def evaluate_grouped_cv(
    input_path: Path,
    output_path: Path,
    *,
    cv_folds: int,
    random_state: int,
) -> pd.DataFrame:
    data = load_bnnupc_dataset(input_path)
    if "simulation_id" not in data.columns:
        raise ValueError("Grouped evaluation requires the simulation_id column.")

    features, target = split_features_and_target(data)
    groups = data.loc[target.index, "simulation_id"]
    if groups.nunique() < cv_folds:
        raise ValueError("The number of simulation groups must be at least cv_folds.")

    predictions = grouped_predictions(
        features,
        target,
        groups,
        cv_folds=cv_folds,
        random_state=random_state,
    )

    scored = data.loc[
        target.index,
        ["qos_class", "scenario", "scheduling_policy", "avg_delay"],
    ].copy()
    scored["true_log_delay"] = target.to_numpy()
    scored["pred_log_delay"] = predictions
    scored["true_delay_ms"] = scored["avg_delay"].to_numpy(dtype=float) * 1000.0
    scored["pred_delay_ms"] = delay_ms_from_log(predictions)

    slice_rows = build_slice_rows(scored, model_name=MODEL_NAME, cv_folds=cv_folds)
    regression = pd.DataFrame(slice_rows)
    regression = regression.loc[
        regression["slice_type"].isin(["overall", "qos_class"])
    ].copy()
    regression.insert(0, "evaluation", "GroupKFold_by_simulation_id")

    sla_rows = build_sla_rows(
        scored,
        model_name=MODEL_NAME,
        cv_folds=cv_folds,
        sla_ms=DEFAULT_SLA_MS,
    )
    sla = pd.DataFrame(sla_rows)
    sla.insert(0, "evaluation", "GroupKFold_by_simulation_id")
    sla["slice_type"] = "sla"
    sla["slice_value"] = sla["qos_class"]

    # Store a single tidy artifact while retaining all underlying metrics.
    result = pd.concat([regression, sla], ignore_index=True, sort=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate BNN-UPC XGBoost with simulation-grouped CV."
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = evaluate_grouped_cv(
        args.input_path,
        args.output_path,
        cv_folds=args.cv_folds,
        random_state=args.random_state,
    )
    print(results.to_string(index=False))
