"""Evaluate the PyTorch MLP on BNN-UPC delay with the same QoS-aware slices.

This is the neural-network counterpart to ``evaluate_bnnupc_qos_slices.py``.
It produces out-of-fold predictions for the PyTorch MLP using the *identical*
cross-validation protocol (KFold, n_splits, shuffle, random_state) and then
reuses the exact same slice and SLA metric functions as the XGBoost evaluator.
This guarantees the per-class numbers are directly comparable: the only thing
that changes between the two reports is the model.

For each fold the MLP needs a small internal validation split for early
stopping. That split is carved out of the training fold only, so no test-fold
information leaks into training.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, train_test_split

# Reuse the MLP architecture and training loop so the per-class evaluation uses
# exactly the same network as train_bnnupc_mlp.py.
from train_bnnupc_mlp import (
    _to_numpy_float,
    build_mlp,  # noqa: F401  (imported for parity / documentation)
    build_preprocessor,
    train_one_split,
)

# Reuse the metric/reporting functions so XGBoost and MLP reports share schema
# and computation. This is what makes the comparison fair.
from evaluate_bnnupc_qos_slices import (
    DEFAULT_SLA_MS,
    QOS_ORDER,  # noqa: F401
    append_results_csv,
    build_sla_rows,
    build_slice_rows,
    delay_ms_from_log,
    load_bnnupc_dataset,
    parse_sla_thresholds,
    split_features_and_target,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "bnnupc_qos_dataset.csv"
DEFAULT_SLICE_OUTPUT_PATH = PROJECT_ROOT / "reports" / "model_results" / "bnnupc_qos_slice_evaluation.csv"
DEFAULT_SLA_OUTPUT_PATH = PROJECT_ROOT / "reports" / "model_results" / "bnnupc_sla_violation_precision.csv"
MODEL_NAME = "PyTorchMLP"


def out_of_fold_mlp_predictions(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    cv_folds: int,
    random_state: int,
    hidden_dims: list[int],
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    patience: int,
    val_size: float,
    device_name: str,
) -> np.ndarray:
    """Return out-of-fold MLP predictions on the log-delay target.

    Matches the XGBoost evaluator's KFold(shuffle, random_state) so the row-wise
    predictions are produced under the same protocol. Within each training fold a
    small validation split drives early stopping.
    """
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    predictions = np.empty(len(features), dtype=float)

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(features), start=1):
        x_train_full = features.iloc[train_idx]
        x_test = features.iloc[test_idx]
        y_train_full = target.iloc[train_idx]

        # Internal validation split for early stopping (train fold only).
        x_fit, x_val, y_fit, y_val = train_test_split(
            x_train_full,
            y_train_full,
            test_size=val_size,
            random_state=random_state + fold_idx,
        )

        preprocessor = build_preprocessor(x_fit)
        x_fit_np = _to_numpy_float(preprocessor.fit_transform(x_fit))
        x_val_np = _to_numpy_float(preprocessor.transform(x_val))
        x_test_np = _to_numpy_float(preprocessor.transform(x_test))
        y_fit_np = y_fit.to_numpy(dtype=np.float32)
        y_val_np = y_val.to_numpy(dtype=np.float32)

        # Train with early stopping on the internal validation split, then the
        # returned model is used to predict the held-out test fold.
        _, _, _, _, _, model = train_one_split(
            x_fit_np,
            y_fit_np,
            x_val_np,
            y_val_np,
            hidden_dims=hidden_dims,
            dropout=dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            random_state=random_state + fold_idx,
            device_name=device_name,
        )

        from train_bnnupc_mlp import predict_with_model

        fold_predictions, _ = predict_with_model(model, x_test_np, device_name)
        predictions[test_idx] = fold_predictions

        print(f"  fold {fold_idx}/{cv_folds} done ({len(test_idx)} test rows)")

    return predictions


def evaluate_mlp_slices(
    data: pd.DataFrame,
    *,
    cv_folds: int,
    random_state: int,
    sla_ms: dict[str, float],
    hidden_dims: list[int],
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    patience: int,
    val_size: float,
    device_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features, target = split_features_and_target(data)
    predictions = out_of_fold_mlp_predictions(
        features,
        target,
        cv_folds=cv_folds,
        random_state=random_state,
        hidden_dims=hidden_dims,
        dropout=dropout,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        epochs=epochs,
        patience=patience,
        val_size=val_size,
        device_name=device_name,
    )

    scored = data.loc[target.index, ["qos_class", "scenario", "scheduling_policy", "avg_delay"]].copy()
    scored["true_log_delay"] = target.to_numpy()
    scored["pred_log_delay"] = predictions
    scored["true_delay_ms"] = scored["avg_delay"].to_numpy(dtype=float) * 1000.0
    scored["pred_delay_ms"] = delay_ms_from_log(predictions)

    slice_rows = build_slice_rows(scored, model_name=MODEL_NAME, cv_folds=cv_folds)
    sla_rows = build_sla_rows(scored, model_name=MODEL_NAME, cv_folds=cv_folds, sla_ms=sla_ms)
    return pd.DataFrame(slice_rows), pd.DataFrame(sla_rows)


def evaluate_bnnupc_mlp_slices(
    input_path: Path,
    slice_output_path: Path,
    sla_output_path: Path,
    cv_folds: int,
    random_state: int,
    sla_ms: dict[str, float],
    hidden_dims: list[int],
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    patience: int,
    val_size: float,
    device_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = load_bnnupc_dataset(input_path)
    slice_df, sla_df = evaluate_mlp_slices(
        data,
        cv_folds=cv_folds,
        random_state=random_state,
        sla_ms=sla_ms,
        hidden_dims=hidden_dims,
        dropout=dropout,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        epochs=epochs,
        patience=patience,
        val_size=val_size,
        device_name=device_name,
    )
    append_results_csv(slice_df, slice_output_path)
    append_results_csv(sla_df, sla_output_path)
    return slice_df, sla_df


def parse_hidden_dims(value: str) -> list[int]:
    dims = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not dims:
        raise argparse.ArgumentTypeError("At least one hidden layer size is required.")
    return dims


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the PyTorch MLP on BNN-UPC delay by QoS class, policy, and SLA precision."
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--slice-output-path", type=Path, default=DEFAULT_SLICE_OUTPUT_PATH)
    parser.add_argument("--sla-output-path", type=Path, default=DEFAULT_SLA_OUTPUT_PATH)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--sla-ms",
        type=parse_sla_thresholds,
        default=DEFAULT_SLA_MS,
        help="SLA violation thresholds in milliseconds as Gold,Silver,Bronze.",
    )
    # MLP hyperparameters mirror train_bnnupc_mlp.py defaults for a faithful comparison.
    parser.add_argument("--hidden-dims", type=parse_hidden_dims, default=parse_hidden_dims("128,64,32"))
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--val-size", type=float, default=0.10)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sla_ms = args.sla_ms if isinstance(args.sla_ms, dict) else DEFAULT_SLA_MS
    print(f"Evaluating {MODEL_NAME} with {args.cv_folds}-fold out-of-fold predictions...")
    slice_results, sla_results = evaluate_bnnupc_mlp_slices(
        input_path=args.input_path,
        slice_output_path=args.slice_output_path,
        sla_output_path=args.sla_output_path,
        cv_folds=args.cv_folds,
        random_state=args.random_state,
        sla_ms=sla_ms,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        val_size=args.val_size,
        device_name=args.device,
    )
    print(f"\nMLP QoS slice evaluation written to: {args.slice_output_path}")
    print(f"MLP SLA violation precision written to: {args.sla_output_path}")
    print("\nPer-class delay MAE/RMSLE:")
    print(
        slice_results.loc[slice_results["slice_type"].eq("qos_class"), [
            "model", "slice_value", "rows", "mae_delay_ms", "rmsle_delay_ms", "r2_log_delay",
        ]].to_string(index=False)
    )
    print("\nSLA trigger precision:")
    print(
        sla_results[[
            "model", "qos_class", "sla_threshold_ms", "actual_violation_rate",
            "precision", "recall", "f1_score",
        ]].to_string(index=False)
    )
