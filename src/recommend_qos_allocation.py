"""Recommend WFQ bandwidth allocations for Gold/Silver/Bronze QoS classes.

The recommender trains a BNN-UPC log-delay model with the same leakage-safe
features used by the XGBoost baseline, then evaluates candidate WFQ profiles by
rewriting the per-class queue weights in a traffic pattern and predicting the
delay each QoS class would experience.

The output answers the Layer 3 allocation question:
"How much bandwidth should be allocated to each QoS class?"
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "bnnupc_qos_dataset.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "reports" / "model_results" / "bnnupc_qos_allocation_recommendations.csv"
TARGET_COLUMN = "log_avg_delay"
QOS_ORDER = ["Gold", "Silver", "Bronze"]
TOS_TO_CLASS = {0: "Gold", 1: "Silver", 2: "Bronze"}
CLASS_TO_TOS = {v: k for k, v in TOS_TO_CLASS.items()}
DEFAULT_PROFILES = [
    "60,30,10",
    "50,40,10",
    "33,33,34",
    "25,65,10",
    "80,15,5",
]
DEFAULT_SLA_MS = {
    "Gold": 20.0,
    "Silver": 50.0,
    "Bronze": 100.0,
}
SLA_PENALTY_WEIGHTS = {
    "Gold": 3.0,
    "Silver": 2.0,
    "Bronze": 1.0,
}

BNNUPC_DROP_COLUMNS = [
    "simulation_id",
    "scenario",
    "avg_delay",
    "jitter",
    "packet_loss_rate",
    "delay_p10",
    "delay_p50",
    "delay_p90",
    "actual_bandwidth",
]


def build_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in features.columns if c not in numeric_cols]

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_cols,
            ),
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


def load_bnnupc_dataset(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset does not exist: {input_path}")
    data = pd.read_csv(input_path)
    data.columns = data.columns.astype(str).str.strip()
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in {input_path}")
    return data


def split_features_and_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    columns_to_drop = [c for c in [TARGET_COLUMN, *BNNUPC_DROP_COLUMNS] if c in data.columns]
    features = data.drop(columns=columns_to_drop)
    target = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")
    mask = target.notna()
    return features.loc[mask].copy(), target.loc[mask].copy()


def build_xgboost_pipeline(features: pd.DataFrame, random_state: int) -> Pipeline:
    try:
        from xgboost import XGBRegressor
    except Exception as exc:
        raise RuntimeError(
            "XGBoost is required for the allocation recommender. "
            "Install project dependencies with `.venv/bin/python -m pip install -r requirements.txt`."
        ) from exc

    model = XGBRegressor(
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
    )
    return Pipeline(steps=[
        ("preprocess", build_preprocessor(features)),
        ("model", model),
    ])


def parse_profile(profile: str) -> dict[str, float]:
    parts = [float(part.strip()) for part in profile.split(",") if part.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Profile '{profile}' must have three comma-separated weights: Gold,Silver,Bronze."
        )
    if any(weight < 0 for weight in parts):
        raise argparse.ArgumentTypeError(f"Profile '{profile}' contains a negative weight.")
    total = sum(parts)
    if total <= 0:
        raise argparse.ArgumentTypeError(f"Profile '{profile}' must have a positive total weight.")
    return dict(zip(QOS_ORDER, parts))


def parse_profiles(values: list[str]) -> list[dict[str, float]]:
    profiles = []
    seen = set()
    for value in values:
        profile = parse_profile(value)
        key = tuple(profile[qos] for qos in QOS_ORDER)
        if key not in seen:
            profiles.append(profile)
            seen.add(key)
    return profiles


def parse_sla_thresholds(value: str) -> dict[str, float]:
    if not value:
        return DEFAULT_SLA_MS.copy()
    parts = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--sla-ms must contain Gold,Silver,Bronze thresholds.")
    if any(threshold <= 0 for threshold in parts):
        raise argparse.ArgumentTypeError("SLA thresholds must be positive.")
    return dict(zip(QOS_ORDER, parts))


def profile_name(profile: dict[str, float]) -> str:
    return "/".join(f"{profile[qos]:g}" for qos in QOS_ORDER)


def prepare_traffic_pattern(
    features: pd.DataFrame,
    *,
    traffic_pattern_path: Path | None,
    sample_rows: int | None,
    random_state: int,
) -> pd.DataFrame:
    if traffic_pattern_path is not None:
        pattern = pd.read_csv(traffic_pattern_path)
        pattern.columns = pattern.columns.astype(str).str.strip()
        missing = [c for c in features.columns if c not in pattern.columns]
        if missing:
            raise ValueError(f"Traffic pattern is missing required feature columns: {missing}")
        pattern = pattern[features.columns].copy()
    else:
        pattern = features.copy()

    if sample_rows is not None and sample_rows > 0 and sample_rows < len(pattern):
        pattern = pattern.sample(n=sample_rows, random_state=random_state)

    if "qos_class" not in pattern.columns or "tos" not in pattern.columns:
        raise ValueError("Traffic pattern must include both 'tos' and 'qos_class'.")

    pattern["tos"] = pd.to_numeric(pattern["tos"], errors="coerce").astype("Int64")
    pattern["qos_class"] = pattern["tos"].map(TOS_TO_CLASS).fillna(pattern["qos_class"])
    return pattern.reset_index(drop=True)


def apply_wfq_profile(pattern: pd.DataFrame, profile: dict[str, float]) -> pd.DataFrame:
    candidate = pattern.copy()
    candidate["scheduling_policy"] = "WFQ"
    candidate["tos_queue_weight"] = candidate["qos_class"].map(profile).astype(float)
    candidate["min_tos_weight"] = candidate["qos_class"].map(profile).astype(float)
    return candidate


def predicted_delay_ms_from_log(predicted_log_delay: np.ndarray) -> np.ndarray:
    return np.exp(predicted_log_delay) * 1000.0


def class_summary(
    candidate: pd.DataFrame,
    predicted_delay_ms: np.ndarray,
    sla_ms: dict[str, float],
) -> dict[str, float | int]:
    scored = candidate[["qos_class"]].copy()
    scored["predicted_delay_ms"] = predicted_delay_ms

    summary: dict[str, float | int] = {}
    weighted_violation_score = 0.0
    weighted_delay_score = 0.0
    feasible = True

    for qos_class in QOS_ORDER:
        class_delays = scored.loc[scored["qos_class"].eq(qos_class), "predicted_delay_ms"]
        if class_delays.empty:
            mean_delay = np.nan
            p90_delay = np.nan
            max_delay = np.nan
            violation_rate = np.nan
            class_count = 0
            feasible = False
        else:
            threshold = sla_ms[qos_class]
            mean_delay = float(class_delays.mean())
            p90_delay = float(class_delays.quantile(0.90))
            max_delay = float(class_delays.max())
            violation_rate = float((class_delays > threshold).mean())
            class_count = int(len(class_delays))
            feasible = feasible and mean_delay <= threshold
            weighted_violation_score += SLA_PENALTY_WEIGHTS[qos_class] * max(0.0, mean_delay / threshold - 1.0)
            weighted_delay_score += SLA_PENALTY_WEIGHTS[qos_class] * (mean_delay / threshold)

        prefix = qos_class.lower()
        summary[f"{prefix}_rows"] = class_count
        summary[f"{prefix}_mean_delay_ms"] = mean_delay
        summary[f"{prefix}_p90_delay_ms"] = p90_delay
        summary[f"{prefix}_max_delay_ms"] = max_delay
        summary[f"{prefix}_sla_ms"] = sla_ms[qos_class]
        summary[f"{prefix}_violation_rate"] = violation_rate

    summary["sla_feasible"] = bool(feasible)
    summary["weighted_violation_score"] = float(weighted_violation_score)
    summary["weighted_delay_score"] = float(weighted_delay_score)
    return summary


def evaluate_profiles(
    model: Pipeline,
    traffic_pattern: pd.DataFrame,
    profiles: list[dict[str, float]],
    sla_ms: dict[str, float],
) -> pd.DataFrame:
    rows = []
    for profile in profiles:
        candidate = apply_wfq_profile(traffic_pattern, profile)
        predicted_log_delay = model.predict(candidate)
        predicted_delay_ms = predicted_delay_ms_from_log(predicted_log_delay)
        row = {
            "run_timestamp": datetime.now().isoformat(timespec="seconds"),
            "profile": profile_name(profile),
            "gold_weight": profile["Gold"],
            "silver_weight": profile["Silver"],
            "bronze_weight": profile["Bronze"],
            "traffic_rows": len(candidate),
            **class_summary(candidate, predicted_delay_ms, sla_ms),
        }
        rows.append(row)

    results = pd.DataFrame(rows)
    results = results.sort_values(
        by=["sla_feasible", "weighted_violation_score", "weighted_delay_score"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    results.insert(1, "rank", np.arange(1, len(results) + 1))
    results.insert(2, "recommended", results["rank"].eq(1))
    return results


def train_delay_model(
    features: pd.DataFrame,
    target: pd.Series,
    random_state: int,
) -> tuple[Pipeline, dict[str, float]]:
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=random_state,
    )
    pipeline = build_xgboost_pipeline(features, random_state=random_state)
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    metrics = {
        "holdout_rmse_log_delay": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "holdout_r2_log_delay": float(r2_score(y_test, predictions)),
    }
    return pipeline, metrics


def recommend_allocations(
    input_path: Path,
    output_path: Path,
    traffic_pattern_path: Path | None,
    profiles: list[dict[str, float]],
    sla_ms: dict[str, float],
    sample_rows: int | None,
    random_state: int,
) -> pd.DataFrame:
    data = load_bnnupc_dataset(input_path)
    features, target = split_features_and_target(data)
    model, model_metrics = train_delay_model(features, target, random_state=random_state)
    traffic_pattern = prepare_traffic_pattern(
        features,
        traffic_pattern_path=traffic_pattern_path,
        sample_rows=sample_rows,
        random_state=random_state,
    )
    recommendations = evaluate_profiles(model, traffic_pattern, profiles, sla_ms)
    for metric, value in model_metrics.items():
        recommendations[metric] = value

    output_path.parent.mkdir(parents=True, exist_ok=True)
    append_to_existing = output_path.exists() and output_path.stat().st_size > 0
    recommendations.to_csv(
        output_path,
        mode="a" if append_to_existing else "w",
        header=not append_to_existing,
        index=False,
    )
    return recommendations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recommend WFQ bandwidth allocations for BNN-UPC QoS classes."
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--traffic-pattern-path",
        type=Path,
        default=None,
        help="Optional CSV containing the traffic pattern to score. Defaults to the processed BNN-UPC features.",
    )
    parser.add_argument(
        "--profile",
        action="append",
        default=[],
        help="Candidate WFQ profile as Gold,Silver,Bronze weights. Repeatable.",
    )
    parser.add_argument(
        "--sla-ms",
        default="20,50,100",
        help="SLA thresholds in milliseconds as Gold,Silver,Bronze.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="Optional number of traffic rows to sample for faster what-if scoring.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    profile_values = args.profile or DEFAULT_PROFILES
    results = recommend_allocations(
        input_path=args.input_path,
        output_path=args.output_path,
        traffic_pattern_path=args.traffic_pattern_path,
        profiles=parse_profiles(profile_values),
        sla_ms=parse_sla_thresholds(args.sla_ms),
        sample_rows=args.sample_rows,
        random_state=args.random_state,
    )
    print(f"Allocation recommendations written to: {args.output_path}")
    display_cols = [
        "rank", "recommended", "profile", "sla_feasible",
        "gold_mean_delay_ms", "silver_mean_delay_ms", "bronze_mean_delay_ms",
        "weighted_violation_score", "weighted_delay_score",
    ]
    print(results[display_cols].to_string(index=False))
