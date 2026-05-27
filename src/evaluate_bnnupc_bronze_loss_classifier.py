"""Evaluate Bronze packet-loss risk as a binary classification task.

The BNN-UPC v1 dataset has near-zero packet loss for Gold and Silver traffic,
so packet-loss modelling is only meaningful for Bronze flows. This script
predicts whether a Bronze flow experiences any loss at all.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from train_baseline import build_preprocessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "bnnupc_qos_dataset.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "reports" / "model_results" / "bnnupc_bronze_loss_classifier.csv"
TARGET_COLUMN = "packet_loss_rate"
QOS_CLASS = "Bronze"

DROP_COLUMNS = [
    "simulation_id",
    "scenario",
    "tos",
    "qos_class",
    "avg_delay",
    "jitter",
    "packet_loss_rate",
    "delay_p10",
    "delay_p50",
    "delay_p90",
    "actual_bandwidth",
    "log_avg_delay",
]


def load_bronze_features(input_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset does not exist: {input_path}")

    data = pd.read_csv(input_path)
    data.columns = data.columns.astype(str).str.strip()
    required = {"qos_class", TARGET_COLUMN}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Input dataset is missing required columns: {missing}")

    bronze = data.loc[data["qos_class"].eq(QOS_CLASS)].copy()
    if bronze.empty:
        raise ValueError(f"No {QOS_CLASS} rows found in {input_path}")

    target = pd.to_numeric(bronze[TARGET_COLUMN], errors="coerce")
    mask = target.notna()
    bronze = bronze.loc[mask].copy()
    target = target.loc[mask].gt(0).astype(int)

    features = bronze.drop(columns=[c for c in DROP_COLUMNS if c in bronze.columns])
    return features, target


def build_model_specs(random_state: int, scale_pos_weight: float) -> list[tuple[str, object]]:
    try:
        from xgboost import XGBClassifier
    except Exception as exc:
        raise RuntimeError(
            "XGBoost is required for this evaluation. Run with the project "
            "virtual environment and installed requirements."
        ) from exc

    xgb_params = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": random_state,
        "n_jobs": -1,
        "verbosity": 0,
    }
    return [
        ("XGBoost_default_threshold", XGBClassifier(**xgb_params)),
        ("XGBoost_class_weighted", XGBClassifier(**xgb_params, scale_pos_weight=scale_pos_weight)),
        (
            "RandomForest_balanced",
            RandomForestClassifier(
                n_estimators=200,
                random_state=random_state,
                n_jobs=-1,
                class_weight="balanced",
            ),
        ),
    ]


def evaluate_model(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    model: object,
    cv_folds: int,
    random_state: int,
    decision_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    probabilities = np.zeros(len(target), dtype=float)
    predictions = np.zeros(len(target), dtype=int)

    for train_idx, test_idx in cv.split(features, target):
        x_train = features.iloc[train_idx]
        x_test = features.iloc[test_idx]
        y_train = target.iloc[train_idx]
        preprocessor = build_preprocessor(x_train, scale_numeric=False)
        pipeline = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", clone(model)),
        ])
        pipeline.fit(x_train, y_train)
        probabilities[test_idx] = pipeline.predict_proba(x_test)[:, 1]
        predictions[test_idx] = (probabilities[test_idx] >= decision_threshold).astype(int)

    return probabilities, predictions


def evaluate_bronze_loss_classifier(
    input_path: Path,
    output_path: Path,
    cv_folds: int,
    random_state: int,
    decision_threshold: float,
) -> pd.DataFrame:
    features, target = load_bronze_features(input_path)
    positives = int(target.sum())
    negatives = int(len(target) - positives)
    if positives == 0:
        raise ValueError("Bronze packet-loss classifier needs at least one positive loss example.")

    scale_pos_weight = negatives / positives
    rows = []
    run_timestamp = datetime.now().isoformat(timespec="seconds")

    for model_name, model in build_model_specs(random_state, scale_pos_weight):
        probabilities, predictions = evaluate_model(
            features,
            target,
            model=model,
            cv_folds=cv_folds,
            random_state=random_state,
            decision_threshold=decision_threshold,
        )
        rows.append({
            "run_timestamp": run_timestamp,
            "model": model_name,
            "target": "packet_loss_rate_gt_0",
            "qos_class": QOS_CLASS,
            "cv_folds": cv_folds,
            "rows": len(target),
            "positive_examples": positives,
            "negative_examples": negatives,
            "positive_rate": positives / len(target),
            "decision_threshold": decision_threshold,
            "scale_pos_weight": scale_pos_weight if "XGBoost_class_weighted" in model_name else "",
            "feature_count": features.shape[1],
            "precision": float(precision_score(target, predictions, zero_division=0)),
            "recall": float(recall_score(target, predictions, zero_division=0)),
            "f1_score": float(f1_score(target, predictions, zero_division=0)),
            "auc_roc": float(roc_auc_score(target, probabilities)),
        })

    result = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate binary packet-loss classification for Bronze BNN-UPC flows."
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = evaluate_bronze_loss_classifier(
        input_path=args.input_path,
        output_path=args.output_path,
        cv_folds=args.cv_folds,
        random_state=args.random_state,
        decision_threshold=args.decision_threshold,
    )
    print(f"Bronze loss classifier results written to: {args.output_path}")
    print(
        results[[
            "model",
            "rows",
            "positive_rate",
            "precision",
            "recall",
            "f1_score",
            "auc_roc",
        ]].to_string(index=False)
    )
