"""Evaluate BNN-UPC QoS targets with per-class and policy slices.

This complements the log-delay evaluator by supporting additional QoS targets
such as jitter and delay_p90. Results are written with a target-aware schema so
different metrics can be appended to one report without mixing column meanings.

The default run evaluates XGBoost on jitter and delay_p90. These targets are
non-negative and heavy-tailed, so the report includes MAE/RMSE on the native
target scale, MAE/RMSE in milliseconds, R2, and RMSLE in milliseconds.
"""

from __future__ import annotations

import argparse
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from train_baseline import build_model_specs, build_preprocessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "bnnupc_qos_dataset.csv"
DEFAULT_SLICE_OUTPUT_PATH = PROJECT_ROOT / "reports" / "model_results" / "bnnupc_metric_slice_evaluation.csv"
DEFAULT_SLA_OUTPUT_PATH = PROJECT_ROOT / "reports" / "model_results" / "bnnupc_metric_sla_precision.csv"
DEFAULT_TARGETS = ["jitter", "delay_p90"]
QOS_ORDER = ["Gold", "Silver", "Bronze"]
OUTCOME_COLUMNS = [
    "avg_delay",
    "log_avg_delay",
    "jitter",
    "packet_loss_rate",
    "delay_p10",
    "delay_p50",
    "delay_p90",
    "actual_bandwidth",
]
IDENTIFIER_COLUMNS = ["simulation_id", "scenario"]


def parse_sla_thresholds(value: str) -> dict[str, float]:
    parts = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("SLA thresholds must be Gold,Silver,Bronze in milliseconds.")
    if any(threshold <= 0 for threshold in parts):
        raise argparse.ArgumentTypeError("SLA thresholds must be positive.")
    return dict(zip(QOS_ORDER, parts))


def append_results_csv(rows: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    append_to_existing = output_path.exists() and output_path.stat().st_size > 0
    rows.to_csv(
        output_path,
        mode="a" if append_to_existing else "w",
        header=not append_to_existing,
        index=False,
    )


def load_bnnupc_dataset(input_path: Path, target_columns: list[str]) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset does not exist: {input_path}")
    data = pd.read_csv(input_path)
    data.columns = data.columns.astype(str).str.strip()
    required = {"qos_class", "scenario", "scheduling_policy", *target_columns}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Input dataset is missing required columns: {missing}")
    return data


def split_features_and_target(
    data: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    columns_to_drop = [c for c in [*IDENTIFIER_COLUMNS, *OUTCOME_COLUMNS] if c in data.columns]
    features = data.drop(columns=columns_to_drop)
    target = pd.to_numeric(data[target_column], errors="coerce")
    mask = target.notna()
    return features.loc[mask].copy(), target.loc[mask].copy()


def model_specs_by_name(random_state: int) -> dict[str, object]:
    specs = {}
    for spec in build_model_specs(random_state):
        if spec.model is not None:
            specs[spec.name] = spec
    return specs


def out_of_fold_predictions(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    model_name: str,
    cv_folds: int,
    random_state: int,
) -> np.ndarray:
    specs = model_specs_by_name(random_state)
    if model_name not in specs:
        available = ", ".join(sorted(specs))
        raise ValueError(f"Unknown or unavailable model '{model_name}'. Available: {available}")

    spec = specs[model_name]
    preprocessor = build_preprocessor(features, scale_numeric=spec.scale_numeric)
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    predictions = np.empty(len(features), dtype=float)

    for train_idx, test_idx in cv.split(features):
        x_train = features.iloc[train_idx]
        x_test = features.iloc[test_idx]
        y_train = target.iloc[train_idx]
        pipeline = Pipeline(steps=[
            ("preprocess", clone(preprocessor)),
            ("model", clone(spec.model)),
        ])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            pipeline.fit(x_train, y_train)
            predictions[test_idx] = pipeline.predict(x_test)

    return predictions


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    if len(y_true) < 2 or np.isclose(np.var(y_true), 0.0):
        return None
    return float(r2_score(y_true, y_pred))


def metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2_score": safe_r2(y_true, y_pred),
    }


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_clipped = np.clip(y_true, a_min=0.0, a_max=None)
    y_pred_clipped = np.clip(y_pred, a_min=0.0, a_max=None)
    return float(np.sqrt(mean_squared_error(np.log1p(y_true_clipped), np.log1p(y_pred_clipped))))


def target_family(target_column: str) -> str:
    if target_column == "jitter":
        return "jitter"
    if target_column.startswith("delay_"):
        return "tail_delay"
    if target_column == "packet_loss_rate":
        return "loss"
    if target_column == "actual_bandwidth":
        return "throughput"
    return "qos_metric"


def build_slice_rows(
    scored: pd.DataFrame,
    *,
    model_name: str,
    target_column: str,
    cv_folds: int,
    unit_scale: float,
    scaled_unit: str,
) -> list[dict[str, object]]:
    rows = []
    slice_specs = [
        ("overall", None),
        ("qos_class", "qos_class"),
        ("scenario", "scenario"),
        ("scheduling_policy", "scheduling_policy"),
    ]
    run_timestamp = datetime.now().isoformat(timespec="seconds")

    for slice_type, column in slice_specs:
        groups = [("all", scored)] if column is None else scored.groupby(column, sort=True)
        for slice_value, group in groups:
            raw_metrics = metric_values(
                group["true_target"].to_numpy(),
                group["pred_target"].to_numpy(),
            )
            scaled_metrics = metric_values(
                group["true_target_scaled"].to_numpy(),
                group["pred_target_scaled"].to_numpy(),
            )
            rows.append({
                "run_timestamp": run_timestamp,
                "model": model_name,
                "target": target_column,
                "target_family": target_family(target_column),
                "cv_folds": cv_folds,
                "slice_type": slice_type,
                "slice_value": slice_value,
                "rows": len(group),
                "target_unit": "native",
                "scaled_unit": scaled_unit,
                "unit_scale": unit_scale,
                "mae": raw_metrics["mae"],
                "rmse": raw_metrics["rmse"],
                "r2_score": raw_metrics["r2_score"],
                "mae_scaled": scaled_metrics["mae"],
                "rmse_scaled": scaled_metrics["rmse"],
                "rmsle_scaled": rmsle(
                    group["true_target_scaled"].to_numpy(),
                    group["pred_target_scaled"].to_numpy(),
                ),
                "r2_score_scaled": scaled_metrics["r2_score"],
            })
    return rows


def build_sla_rows(
    scored: pd.DataFrame,
    *,
    model_name: str,
    target_column: str,
    cv_folds: int,
    sla_ms: dict[str, float],
) -> list[dict[str, object]]:
    rows = []
    run_timestamp = datetime.now().isoformat(timespec="seconds")

    for qos_class in QOS_ORDER:
        group = scored.loc[scored["qos_class"].eq(qos_class)].copy()
        if group.empty:
            continue
        threshold = sla_ms[qos_class]
        actual_violation = group["true_target_scaled"].gt(threshold)
        predicted_violation = group["pred_target_scaled"].gt(threshold)
        rows.append({
            "run_timestamp": run_timestamp,
            "model": model_name,
            "target": target_column,
            "target_family": target_family(target_column),
            "cv_folds": cv_folds,
            "qos_class": qos_class,
            "sla_threshold_ms": threshold,
            "rows": len(group),
            "actual_violations": int(actual_violation.sum()),
            "predicted_violations": int(predicted_violation.sum()),
            "actual_violation_rate": float(actual_violation.mean()),
            "predicted_violation_rate": float(predicted_violation.mean()),
            "precision": float(precision_score(actual_violation, predicted_violation, zero_division=0)),
            "recall": float(recall_score(actual_violation, predicted_violation, zero_division=0)),
            "f1_score": float(f1_score(actual_violation, predicted_violation, zero_division=0)),
        })
    return rows


def evaluate_target(
    data: pd.DataFrame,
    *,
    target_column: str,
    model_name: str,
    cv_folds: int,
    random_state: int,
    unit_scale: float,
    scaled_unit: str,
    sla_ms: dict[str, float] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features, target = split_features_and_target(data, target_column)
    predictions = out_of_fold_predictions(
        features,
        target,
        model_name=model_name,
        cv_folds=cv_folds,
        random_state=random_state,
    )

    scored = data.loc[target.index, ["qos_class", "scenario", "scheduling_policy"]].copy()
    scored["true_target"] = target.to_numpy(dtype=float)
    scored["pred_target"] = predictions
    scored["true_target_scaled"] = scored["true_target"] * unit_scale
    scored["pred_target_scaled"] = scored["pred_target"] * unit_scale

    slice_rows = build_slice_rows(
        scored,
        model_name=model_name,
        target_column=target_column,
        cv_folds=cv_folds,
        unit_scale=unit_scale,
        scaled_unit=scaled_unit,
    )
    sla_rows = [] if sla_ms is None else build_sla_rows(
        scored,
        model_name=model_name,
        target_column=target_column,
        cv_folds=cv_folds,
        sla_ms=sla_ms,
    )
    return pd.DataFrame(slice_rows), pd.DataFrame(sla_rows)


def evaluate_bnnupc_metric_slices(
    input_path: Path,
    slice_output_path: Path,
    sla_output_path: Path,
    target_columns: list[str],
    model_names: list[str],
    cv_folds: int,
    random_state: int,
    unit_scale: float,
    scaled_unit: str,
    sla_ms: dict[str, float] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = load_bnnupc_dataset(input_path, target_columns)
    all_slice_results = []
    all_sla_results = []

    for target_column in target_columns:
        for model_name in model_names:
            slice_results, sla_results = evaluate_target(
                data,
                target_column=target_column,
                model_name=model_name,
                cv_folds=cv_folds,
                random_state=random_state,
                unit_scale=unit_scale,
                scaled_unit=scaled_unit,
                sla_ms=sla_ms,
            )
            all_slice_results.append(slice_results)
            if not sla_results.empty:
                all_sla_results.append(sla_results)

    slice_df = pd.concat(all_slice_results, ignore_index=True)
    sla_df = pd.concat(all_sla_results, ignore_index=True) if all_sla_results else pd.DataFrame()
    append_results_csv(slice_df, slice_output_path)
    if not sla_df.empty:
        append_results_csv(sla_df, sla_output_path)
    return slice_df, sla_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate BNN-UPC QoS targets by class, scenario, policy, and optional SLA precision."
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--slice-output-path", type=Path, default=DEFAULT_SLICE_OUTPUT_PATH)
    parser.add_argument("--sla-output-path", type=Path, default=DEFAULT_SLA_OUTPUT_PATH)
    parser.add_argument(
        "--target-column",
        action="append",
        default=[],
        help="QoS target to evaluate. Repeatable. Defaults to jitter and delay_p90.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Model to evaluate. Repeatable. Defaults to XGBRegressor.",
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--unit-scale",
        type=float,
        default=1000.0,
        help="Scale native target values for readable reporting. Seconds to milliseconds uses 1000.",
    )
    parser.add_argument("--scaled-unit", default="ms")
    parser.add_argument(
        "--sla-ms",
        type=parse_sla_thresholds,
        default=None,
        help="Optional SLA violation thresholds in scaled milliseconds as Gold,Silver,Bronze.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    targets = args.target_column or DEFAULT_TARGETS
    models = args.model or ["XGBRegressor"]
    slice_results, sla_results = evaluate_bnnupc_metric_slices(
        input_path=args.input_path,
        slice_output_path=args.slice_output_path,
        sla_output_path=args.sla_output_path,
        target_columns=targets,
        model_names=models,
        cv_folds=args.cv_folds,
        random_state=args.random_state,
        unit_scale=args.unit_scale,
        scaled_unit=args.scaled_unit,
        sla_ms=args.sla_ms,
    )
    print(f"Metric slice evaluation written to: {args.slice_output_path}")
    if not sla_results.empty:
        print(f"Metric SLA precision written to: {args.sla_output_path}")
    print("\nPer-class scaled MAE/RMSE/R2:")
    print(
        slice_results.loc[slice_results["slice_type"].eq("qos_class"), [
            "model", "target", "slice_value", "rows",
            "mae_scaled", "rmse_scaled", "rmsle_scaled", "r2_score",
        ]].to_string(index=False)
    )
    if not sla_results.empty:
        print("\nSLA trigger precision:")
        print(
            sla_results[[
                "model", "target", "qos_class", "sla_threshold_ms",
                "actual_violation_rate", "precision", "recall", "f1_score",
            ]].to_string(index=False)
        )
