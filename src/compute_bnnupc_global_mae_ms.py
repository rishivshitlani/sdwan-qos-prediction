"""Re-evaluate the BNN-UPC global model comparison (thesis Table 5.2) with MAE
also reported in milliseconds on the real delay scale.

Table 5.2 reports MAE/RMSE on `log_avg_delay`, the target the models were
actually trained on, because raw delay is heavy-tailed. That number isn't
directly interpretable as a delay error. This script reuses the exact same
held-out test split (test_size, random_state) as train_baseline.py and
train_bnnupc_mlp.py, so the log-scale MAE reproduces the existing thesis
numbers as a sanity check, then back-transforms the same held-out predictions
to milliseconds (avg_delay is stored in seconds, so ms = exp(log) * 1000) and
reports MAE on that scale, directly comparable to the per-class Table 5.3.

XGBoost and PyTorch are scored in separate subprocesses because loading both
native-threading runtimes in one interpreter can conflict (see
compute_model_significance.py, which uses the same pattern).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from train_baseline import build_model_specs, build_preprocessor
from evaluate_bnnupc_qos_slices import delay_ms_from_log, load_bnnupc_dataset, split_features_and_target
from train_bnnupc_mlp import (
    _to_numpy_float,
    build_preprocessor as build_mlp_preprocessor,
    predict_with_model,
    train_one_split as train_mlp_split,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "bnnupc_qos_dataset.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "reports" / "model_results" / "bnnupc_global_results_ms.csv"
SKLEARN_MODEL_ORDER = ["LinearRegression", "SVR_rbf", "RandomForestRegressor", "XGBRegressor"]
MLP_HIDDEN_DIMS = [128, 64, 32]
MLP_DROPOUT = 0.10
MLP_LEARNING_RATE = 1e-3
MLP_WEIGHT_DECAY = 1e-4
MLP_BATCH_SIZE = 256
MLP_EPOCHS = 200
MLP_PATIENCE = 20


def build_metrics_row(model_name: str, y_test_log: np.ndarray, preds_log: np.ndarray) -> dict[str, object]:
    y_test_ms = delay_ms_from_log(y_test_log)
    preds_ms = delay_ms_from_log(preds_log)
    return {
        "model": model_name,
        "test_rows": int(len(y_test_log)),
        "mae_log_delay": float(mean_absolute_error(y_test_log, preds_log)),
        "rmse_log_delay": float(np.sqrt(mean_squared_error(y_test_log, preds_log))),
        "r2_log_delay": float(r2_score(y_test_log, preds_log)),
        "mae_delay_ms": float(mean_absolute_error(y_test_ms, preds_ms)),
        "rmse_delay_ms": float(np.sqrt(mean_squared_error(y_test_ms, preds_ms))),
    }


def score_sklearn_model(
    model_name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int,
) -> dict[str, object]:
    specs = {s.name: s for s in build_model_specs(random_state) if s.model is not None}
    if model_name not in specs:
        raise ValueError(f"Unknown or unavailable model '{model_name}'.")
    spec = specs[model_name]
    pipeline = Pipeline(steps=[
        ("preprocess", build_preprocessor(x_train, scale_numeric=spec.scale_numeric)),
        ("model", clone(spec.model)),
    ])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        pipeline.fit(x_train, y_train)
        preds_log = pipeline.predict(x_test)
    return build_metrics_row(model_name, y_test.to_numpy(dtype=float), preds_log)


def score_mlp(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int,
) -> dict[str, object]:
    # Mirrors train_bnnupc_mlp.train_bnnupc_mlp: a 90/10 fit/val split inside
    # the training portion drives early stopping, matching the thesis protocol.
    x_fit, x_val, y_fit, y_val = train_test_split(x_train, y_train, test_size=0.10, random_state=random_state)
    preprocessor = build_mlp_preprocessor(x_fit)
    x_fit_np = _to_numpy_float(preprocessor.fit_transform(x_fit))
    x_val_np = _to_numpy_float(preprocessor.transform(x_val))
    x_test_np = _to_numpy_float(preprocessor.transform(x_test))

    _, _, _, _, _, model = train_mlp_split(
        x_fit_np, y_fit.to_numpy(dtype=np.float32),
        x_val_np, y_val.to_numpy(dtype=np.float32),
        hidden_dims=MLP_HIDDEN_DIMS, dropout=MLP_DROPOUT,
        learning_rate=MLP_LEARNING_RATE, weight_decay=MLP_WEIGHT_DECAY,
        batch_size=MLP_BATCH_SIZE, epochs=MLP_EPOCHS, patience=MLP_PATIENCE,
        random_state=random_state, device_name="auto",
    )
    preds_log, _ = predict_with_model(model, x_test_np, "auto")
    return build_metrics_row("PyTorchMLP", y_test.to_numpy(dtype=float), preds_log)


def run_mlp_subprocess(
    input_path: Path,
    output_path: Path,
    test_size: float,
    random_state: int,
) -> dict[str, object]:
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--input-path", str(input_path),
        "--test-size", str(test_size),
        "--random-state", str(random_state),
        "--score-mlp-only",
        "--score-output-path", str(output_path),
    ]
    subprocess.run(cmd, check=True)
    return pd.read_csv(output_path).iloc[0].to_dict()


def compare(input_path: Path, output_path: Path, test_size: float, random_state: int) -> pd.DataFrame:
    data = load_bnnupc_dataset(input_path)
    features, target = split_features_and_target(data)
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state,
    )

    rows = []
    for model_name in SKLEARN_MODEL_ORDER:
        row = score_sklearn_model(model_name, x_train, y_train, x_test, y_test, random_state)
        print(f"  {model_name}: mae_log={row['mae_log_delay']:.4f}  mae_ms={row['mae_delay_ms']:.2f}", flush=True)
        rows.append(row)

    with tempfile.TemporaryDirectory(prefix="bnnupc_global_mae_ms_") as tmpdir:
        mlp_output = Path(tmpdir) / "mlp_row.csv"
        mlp_row = run_mlp_subprocess(input_path, mlp_output, test_size, random_state)
    print(f"  PyTorchMLP: mae_log={mlp_row['mae_log_delay']:.4f}  mae_ms={mlp_row['mae_delay_ms']:.2f}", flush=True)
    rows.append(mlp_row)

    result = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"\nWritten to: {output_path}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-evaluate BNN-UPC global results (Table 5.2) with MAE in milliseconds."
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--score-mlp-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--score-output-path", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.score_mlp_only:
        if args.score_output_path is None:
            raise SystemExit("--score-output-path is required with --score-mlp-only")
        data = load_bnnupc_dataset(args.input_path)
        features, target = split_features_and_target(data)
        x_train, x_test, y_train, y_test = train_test_split(
            features, target, test_size=args.test_size, random_state=args.random_state,
        )
        row = score_mlp(x_train, y_train, x_test, y_test, args.random_state)
        pd.DataFrame([row]).to_csv(args.score_output_path, index=False)
    else:
        compare(args.input_path, args.output_path, args.test_size, args.random_state)
