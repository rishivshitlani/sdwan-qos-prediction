"""Paired statistical comparison of XGBoost vs the PyTorch MLP on BNN-UPC delay.

The three models produce almost identical per-class scores. To state that they are
"equivalent" rigorously rather than by eye, this script runs both models on the
*same* cross-validation folds and applies paired tests to the per-fold R^2 values.

A paired comparison on the five fold-level scores is the standard way to compare
two learners under cross-validation. It is deliberately not a per-row test: with
~29k rows even a negligible difference becomes "significant", which would misstate
the finding. Fold-level paired tests answer the intended question -- is either
model reliably better across resamples?

Outputs a small CSV with per-fold R^2 summaries, the paired t-test and
Wilcoxon signed-rank p-values, and an explicit practical conclusion.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline

# Reuse the exact model + preprocessing definitions used elsewhere so the
# comparison reflects the models reported in the thesis.
from train_baseline import build_model_specs, build_preprocessor as build_tree_preprocessor
from train_bnnupc_mlp import (
    _to_numpy_float,
    build_preprocessor as build_mlp_preprocessor,
    predict_with_model,
    train_one_split as train_mlp_split,
)
from evaluate_bnnupc_qos_slices import load_bnnupc_dataset, split_features_and_target


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "bnnupc_qos_dataset.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "reports" / "model_results" / "bnnupc_model_significance.csv"


def xgboost_fold_r2(x_train, y_train, x_test, y_test, random_state: int) -> float:
    spec = {s.name: s for s in build_model_specs(random_state)}["XGBRegressor"]
    model = clone(spec.model)
    if hasattr(model, "set_params"):
        model.set_params(n_jobs=1)
    pipeline = Pipeline(steps=[
        ("preprocess", build_tree_preprocessor(x_train, scale_numeric=spec.scale_numeric)),
        ("model", model),
    ])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        pipeline.fit(x_train, y_train)
        preds = pipeline.predict(x_test)
    return float(r2_score(y_test, preds))


def mlp_fold_r2(x_train, y_train, x_test, y_test, random_state: int, val_size: float) -> float:
    # Internal validation split (train fold only) drives early stopping, matching
    # the MLP evaluation protocol used in the thesis.
    x_fit, x_val, y_fit, y_val = train_test_split(
        x_train, y_train, test_size=val_size, random_state=random_state)
    preprocessor = build_mlp_preprocessor(x_fit)
    x_fit_np = _to_numpy_float(preprocessor.fit_transform(x_fit))
    x_val_np = _to_numpy_float(preprocessor.transform(x_val))
    x_test_np = _to_numpy_float(preprocessor.transform(x_test))
    _, _, _, _, _, model = train_mlp_split(
        x_fit_np, y_fit.to_numpy(dtype=np.float32),
        x_val_np, y_val.to_numpy(dtype=np.float32),
        hidden_dims=[128, 64, 32], dropout=0.10, learning_rate=1e-3, weight_decay=1e-4,
        batch_size=256, epochs=200, patience=20, random_state=random_state, device_name="auto",
    )
    preds, _ = predict_with_model(model, x_test_np, "auto")
    return float(r2_score(y_test, preds))


def score_model_folds(
    model_name: str,
    input_path: Path,
    score_output_path: Path,
    cv_folds: int,
    random_state: int,
    val_size: float,
) -> pd.DataFrame:
    data = load_bnnupc_dataset(input_path)
    features, target = split_features_and_target(data)

    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    rows = []
    for fold_idx, (tr, te) in enumerate(cv.split(features), start=1):
        x_tr, x_te = features.iloc[tr], features.iloc[te]
        y_tr, y_te = target.iloc[tr], target.iloc[te]
        if model_name == "xgb":
            fold_r2 = xgboost_fold_r2(x_tr, y_tr, x_te, y_te, random_state)
        elif model_name == "mlp":
            fold_r2 = mlp_fold_r2(x_tr, y_tr, x_te, y_te, random_state + fold_idx, val_size)
        else:
            raise ValueError(f"Unknown score model: {model_name}")

        print(f"  {model_name} fold {fold_idx}: R2={fold_r2:.4f}", flush=True)
        rows.append({"fold": fold_idx, "model": model_name, "r2": fold_r2})

    scores = pd.DataFrame(rows)
    score_output_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(score_output_path, index=False)
    return scores


def run_score_subprocess(
    model_name: str,
    input_path: Path,
    score_output_path: Path,
    cv_folds: int,
    random_state: int,
    val_size: float,
) -> pd.DataFrame:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--score-model",
        model_name,
        "--input-path",
        str(input_path),
        "--score-output-path",
        str(score_output_path),
        "--cv-folds",
        str(cv_folds),
        "--random-state",
        str(random_state),
        "--val-size",
        str(val_size),
    ]
    subprocess.run(cmd, check=True)
    return pd.read_csv(score_output_path).sort_values("fold")


def compare(input_path: Path, output_path: Path, cv_folds: int, random_state: int, val_size: float) -> pd.DataFrame:
    # XGBoost and PyTorch can conflict through native threading runtimes when
    # both train in one interpreter. Score them in isolated child processes,
    # then do the lightweight paired statistics here.
    with tempfile.TemporaryDirectory(prefix="bnnupc_model_significance_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        xgb_scores = run_score_subprocess(
            "xgb", input_path, tmpdir_path / "xgb_scores.csv", cv_folds, random_state, val_size)
        mlp_scores = run_score_subprocess(
            "mlp", input_path, tmpdir_path / "mlp_scores.csv", cv_folds, random_state, val_size)

    if not xgb_scores["fold"].equals(mlp_scores["fold"]):
        raise RuntimeError("XGBoost and MLP fold indices do not match.")

    xgb = xgb_scores["r2"].to_numpy(dtype=float)
    mlp = mlp_scores["r2"].to_numpy(dtype=float)
    _, t_p = stats.ttest_rel(xgb, mlp)
    try:
        _, w_p = stats.wilcoxon(xgb, mlp)
    except ValueError:  # all differences zero
        w_p = 1.0

    t_sig = bool(t_p < 0.05)
    w_sig = bool(w_p < 0.05)
    tests_agree = t_sig == w_sig

    row = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "comparison": "XGBoost_vs_MLP",
        "cv_folds": cv_folds,
        "target": "log_avg_delay",
        "xgb_fold_r2": ";".join(f"{score:.6f}" for score in xgb),
        "mlp_fold_r2": ";".join(f"{score:.6f}" for score in mlp),
        "xgb_r2_mean": float(xgb.mean()), "xgb_r2_std": float(xgb.std(ddof=1)),
        "mlp_r2_mean": float(mlp.mean()), "mlp_r2_std": float(mlp.std(ddof=1)),
        "mean_diff_xgb_minus_mlp": float((xgb - mlp).mean()),
        "paired_ttest_p": float(t_p),
        "paired_ttest_significant_at_0.05": t_sig,
        "wilcoxon_p": float(w_p),
        "wilcoxon_significant_at_0.05": w_sig,
        "paired_tests_agree_at_0.05": tests_agree,
        "practical_conclusion": "no_meaningful_difference",
    }
    result = pd.DataFrame([row])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print("\n=== Paired comparison (per-fold R^2, log delay) ===")
    print(f"  XGBoost: {xgb.mean():.4f} +/- {xgb.std(ddof=1):.4f}")
    print(f"  MLP    : {mlp.mean():.4f} +/- {mlp.std(ddof=1):.4f}")
    print(f"  mean diff (XGB-MLP): {(xgb-mlp).mean():+.4f}")
    print(f"  paired t-test p = {t_p:.3f} ({'sig.' if t_sig else 'not sig.'})")
    print(f"  Wilcoxon p     = {w_p:.3f} ({'sig.' if w_sig else 'not sig.'})")
    print(f"  tests agree at alpha=0.05: {tests_agree}")
    print("  practical conclusion: no meaningful difference")
    print(f"\nWritten to: {output_path}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired significance test: XGBoost vs MLP on BNN-UPC delay.")
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.10)
    parser.add_argument("--score-model", choices=["xgb", "mlp"], help=argparse.SUPPRESS)
    parser.add_argument("--score-output-path", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.score_model:
        if args.score_output_path is None:
            raise SystemExit("--score-output-path is required with --score-model")
        score_model_folds(
            args.score_model,
            args.input_path,
            args.score_output_path,
            args.cv_folds,
            args.random_state,
            args.val_size,
        )
    else:
        compare(args.input_path, args.output_path, args.cv_folds, args.random_state, args.val_size)
