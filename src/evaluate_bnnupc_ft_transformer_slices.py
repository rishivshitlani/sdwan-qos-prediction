"""Evaluate the FT-Transformer on BNN-UPC delay with QoS-aware slices.

This is the FT-Transformer counterpart to ``evaluate_bnnupc_mlp_slices.py`` and
``evaluate_bnnupc_qos_slices.py``. It produces out-of-fold predictions using the
same outer cross-validation protocol (KFold, n_splits, shuffle, random_state) and
reuses the exact same per-class slice and SLA metric functions as the XGBoost and
MLP evaluators. The per-class numbers are therefore directly comparable while the
model-specific training details stay inside each fold.

Within each fold a small validation split (carved from the training fold only)
drives early stopping; no test-fold information leaks into training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from train_bnnupc_ft_transformer import (
    cat_cardinalities,
    fit_preprocessor,
    predict_with_model,
    train_one_split,
    transform,
)
from evaluate_bnnupc_qos_slices import (
    append_results_csv,
    build_sla_rows,
    build_slice_rows,
    delay_ms_from_log,
    load_bnnupc_dataset,
    parse_sla_thresholds,
    split_features_and_target,
)
from sdwan_qos.config import BNNUPC_PROCESSED_DATASET, EVAL_SLA_MS, REPORTS_DIR


DEFAULT_INPUT_PATH = BNNUPC_PROCESSED_DATASET
DEFAULT_SLICE_OUTPUT_PATH = REPORTS_DIR / "bnnupc_qos_slice_evaluation.csv"
DEFAULT_SLA_OUTPUT_PATH = REPORTS_DIR / "bnnupc_sla_violation_precision.csv"
MODEL_NAME = "FTTransformer"


def out_of_fold_predictions(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    cv_folds: int,
    random_state: int,
    val_size: float,
    hp: dict,
) -> np.ndarray:
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    predictions = np.empty(len(features), dtype=float)

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(features), start=1):
        x_train_full = features.iloc[train_idx]
        x_test = features.iloc[test_idx]
        y_train_full = target.iloc[train_idx]

        x_fit, x_val, y_fit, y_val = train_test_split(
            x_train_full, y_train_full, test_size=val_size, random_state=random_state + fold_idx)

        state = fit_preprocessor(x_fit)
        xn_fit, xc_fit = transform(state, x_fit)
        xn_val, xc_val = transform(state, x_val)
        xn_test, xc_test = transform(state, x_test)

        _, _, _, _, model = train_one_split(
            xn_fit, xc_fit, y_fit.to_numpy(dtype=np.float32),
            xn_val, xc_val, y_val.to_numpy(dtype=np.float32),
            cat_cards=cat_cardinalities(state), random_state=random_state + fold_idx, **hp,
        )
        predictions[test_idx] = predict_with_model(model, xn_test, xc_test, hp["device_name"])
        print(f"  fold {fold_idx}/{cv_folds} done ({len(test_idx)} test rows)", flush=True)

    return predictions


def evaluate_bnnupc_ft_transformer_slices(
    input_path: Path, slice_output_path: Path, sla_output_path: Path, *,
    cv_folds: int, random_state: int, sla_ms: dict, val_size: float, hp: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = load_bnnupc_dataset(input_path)
    features, target = split_features_and_target(data)
    predictions = out_of_fold_predictions(
        features, target, cv_folds=cv_folds, random_state=random_state, val_size=val_size, hp=hp)

    scored = data.loc[target.index, ["qos_class", "scenario", "scheduling_policy", "avg_delay"]].copy()
    scored["true_log_delay"] = target.to_numpy()
    scored["pred_log_delay"] = predictions
    scored["true_delay_ms"] = scored["avg_delay"].to_numpy(dtype=float) * 1000.0
    scored["pred_delay_ms"] = delay_ms_from_log(predictions)

    slice_df = pd.DataFrame(build_slice_rows(scored, model_name=MODEL_NAME, cv_folds=cv_folds))
    sla_df = pd.DataFrame(build_sla_rows(scored, model_name=MODEL_NAME, cv_folds=cv_folds, sla_ms=sla_ms))
    append_results_csv(slice_df, slice_output_path)
    append_results_csv(sla_df, sla_output_path)
    return slice_df, sla_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the FT-Transformer on BNN-UPC delay by QoS class, policy, and SLA precision.")
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--slice-output-path", type=Path, default=DEFAULT_SLICE_OUTPUT_PATH)
    parser.add_argument("--sla-output-path", type=Path, default=DEFAULT_SLA_OUTPUT_PATH)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--sla-ms", type=parse_sla_thresholds, default=EVAL_SLA_MS)
    parser.add_argument("--val-size", type=float, default=0.10)
    parser.add_argument("--d-token", type=int, default=64)
    parser.add_argument("--n-blocks", type=int, default=3)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--ffn-factor", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sla_ms = args.sla_ms if isinstance(args.sla_ms, dict) else EVAL_SLA_MS
    hp = {
        "d_token": args.d_token, "n_blocks": args.n_blocks, "n_heads": args.n_heads,
        "ffn_factor": args.ffn_factor, "dropout": args.dropout,
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "batch_size": args.batch_size, "epochs": args.epochs, "patience": args.patience,
        "device_name": args.device,
    }
    print(f"Evaluating {MODEL_NAME} with {args.cv_folds}-fold out-of-fold predictions...")
    slice_results, sla_results = evaluate_bnnupc_ft_transformer_slices(
        input_path=args.input_path, slice_output_path=args.slice_output_path,
        sla_output_path=args.sla_output_path, cv_folds=args.cv_folds,
        random_state=args.random_state, sla_ms=sla_ms, val_size=args.val_size, hp=hp,
    )
    print(f"\nFT-Transformer QoS slice evaluation written to: {args.slice_output_path}")
    print(f"FT-Transformer SLA violation precision written to: {args.sla_output_path}")
    print("\nPer-class delay MAE/RMSLE:")
    print(slice_results.loc[slice_results["slice_type"].eq("qos_class"), [
        "model", "slice_value", "rows", "mae_delay_ms", "rmsle_delay_ms", "r2_log_delay",
    ]].to_string(index=False))
    print("\nSLA trigger precision:")
    print(sla_results[[
        "model", "qos_class", "sla_threshold_ms", "actual_violation_rate",
        "precision", "recall", "f1_score",
    ]].to_string(index=False))
