"""Train a PyTorch feedforward MLP on the BNN-UPC QoS dataset.

This uses the same raw feature set as the BNN-UPC XGBoost log-delay baseline:
identifier columns and delay/loss outcome columns are dropped, and the target is
``log_avg_delay``. Numeric features are median-imputed and scaled; categorical
features are most-frequent-imputed and one-hot encoded.
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "bnnupc_qos_dataset.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "reports" / "model_results" / "bnnupc_mlp_log_delay_results.csv"
TARGET_COLUMN = "log_avg_delay"
DATASET_NAME = "bnnupc_qos"

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
                Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
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


def load_features_and_target(
    input_path: Path,
    target_column: str,
    drop_columns: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset does not exist: {input_path}")

    data = pd.read_csv(input_path)
    data.columns = data.columns.astype(str).str.strip()
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in {input_path}")

    columns_to_drop = [c for c in [target_column, *drop_columns] if c in data.columns]
    features = data.drop(columns=columns_to_drop)
    target = pd.to_numeric(data[target_column], errors="coerce")
    mask = target.notna()
    return features.loc[mask].copy(), target.loc[mask].copy()


def _import_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyTorch is not installed. Install dependencies with "
            "`.venv/bin/python -m pip install -r requirements.txt`."
        ) from exc
    return torch, nn, DataLoader, TensorDataset


def set_reproducible_seed(torch, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_mlp(nn, input_dim: int, hidden_dims: list[int], dropout: float):
    layers = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend([
            nn.Linear(prev_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        ])
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, 1))
    return nn.Sequential(*layers)


def _to_numpy_float(data) -> np.ndarray:
    if hasattr(data, "toarray"):
        data = data.toarray()
    return np.asarray(data, dtype=np.float32)


def train_one_split(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    *,
    hidden_dims: list[int],
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    patience: int,
    random_state: int,
    device_name: str,
) -> tuple[np.ndarray, int, float, float, float, object]:
    torch, nn, DataLoader, TensorDataset = _import_torch()
    set_reproducible_seed(torch, random_state)

    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_mlp(nn, x_train.shape[1], hidden_dims, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    train_ds = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(y_train.reshape(-1, 1).astype(np.float32)),
    )
    generator = torch.Generator()
    generator.manual_seed(random_state)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator)

    x_eval_tensor = torch.from_numpy(x_eval).to(device)
    y_eval_tensor = torch.from_numpy(y_eval.reshape(-1, 1).astype(np.float32)).to(device)

    best_state = None
    best_eval_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    t0 = perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            eval_loss = float(loss_fn(model(x_eval_tensor), y_eval_tensor).item())

        if eval_loss < best_eval_loss - 1e-6:
            best_eval_loss = eval_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    training_time_sec = perf_counter() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    t1 = perf_counter()
    with torch.no_grad():
        predictions = model(x_eval_tensor).detach().cpu().numpy().reshape(-1)
    inference_time_sec = perf_counter() - t1

    return predictions, best_epoch, best_eval_loss, training_time_sec, inference_time_sec, model


def predict_with_model(model, x_eval: np.ndarray, device_name: str) -> tuple[np.ndarray, float]:
    torch, _, _, _ = _import_torch()
    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    x_eval_tensor = torch.from_numpy(x_eval).to(device)

    t0 = perf_counter()
    with torch.no_grad():
        predictions = model(x_eval_tensor).detach().cpu().numpy().reshape(-1)
    inference_time_sec = perf_counter() - t0
    return predictions, inference_time_sec


def metric_row(
    predictions: np.ndarray,
    y_true: np.ndarray,
    *,
    training_time_sec: float,
    inference_time_sec: float,
    train_rows: int,
    test_rows: int,
    raw_feature_count: int,
    encoded_feature_count: int,
    best_epoch: int,
    best_val_loss: float,
    cv_folds: int,
    cv_metrics: dict[str, float | int | None],
    dataset_name: str,
    target_column: str,
) -> dict[str, object]:
    return {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": dataset_name,
        "model": "PyTorchMLP",
        "target": target_column,
        "status": "ok",
        "error_message": "",
        "train_rows": train_rows,
        "test_rows": test_rows,
        "feature_count": raw_feature_count,
        "encoded_feature_count": encoded_feature_count,
        "mae": mean_absolute_error(y_true, predictions),
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "r2_score": r2_score(y_true, predictions),
        **cv_metrics,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "training_time_sec": training_time_sec,
        "inference_time_sec": inference_time_sec,
        "inference_time_ms_per_row": inference_time_sec / test_rows * 1000,
    }


def run_cv(
    features: pd.DataFrame,
    target: pd.Series,
    preprocessor: ColumnTransformer,
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
    device_name: str,
) -> dict[str, float | int | None]:
    if cv_folds <= 1:
        return {
            "cv_folds": cv_folds,
            "cv_mae_mean": None, "cv_mae_std": None,
            "cv_rmse_mean": None, "cv_rmse_std": None,
            "cv_r2_mean": None, "cv_r2_std": None,
        }

    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    maes, rmses, r2s = [], [], []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(features), start=1):
        x_fold_train = features.iloc[train_idx]
        x_fold_val = features.iloc[val_idx]
        y_fold_train = target.iloc[train_idx].to_numpy(dtype=np.float32)
        y_fold_val = target.iloc[val_idx].to_numpy(dtype=np.float32)

        fold_preprocessor = clone(preprocessor)
        x_fold_train_np = _to_numpy_float(fold_preprocessor.fit_transform(x_fold_train))
        x_fold_val_np = _to_numpy_float(fold_preprocessor.transform(x_fold_val))

        predictions, _, _, _, _, _ = train_one_split(
            x_fold_train_np,
            y_fold_train,
            x_fold_val_np,
            y_fold_val,
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
        maes.append(mean_absolute_error(y_fold_val, predictions))
        rmses.append(float(np.sqrt(mean_squared_error(y_fold_val, predictions))))
        r2s.append(r2_score(y_fold_val, predictions))

    return {
        "cv_folds": cv_folds,
        "cv_mae_mean": float(np.mean(maes)), "cv_mae_std": float(np.std(maes)),
        "cv_rmse_mean": float(np.mean(rmses)), "cv_rmse_std": float(np.std(rmses)),
        "cv_r2_mean": float(np.mean(r2s)), "cv_r2_std": float(np.std(r2s)),
    }


def train_bnnupc_mlp(
    input_path: Path,
    output_path: Path,
    test_size: float,
    random_state: int,
    cv_folds: int,
    hidden_dims: list[int],
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    patience: int,
    device_name: str,
) -> pd.DataFrame:
    features, target = load_features_and_target(input_path, TARGET_COLUMN, BNNUPC_DROP_COLUMNS)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )

    x_fit, x_val, y_fit, y_val = train_test_split(
        x_train,
        y_train,
        test_size=0.10,
        random_state=random_state,
    )

    preprocessor = build_preprocessor(x_fit)
    x_fit_np = _to_numpy_float(preprocessor.fit_transform(x_fit))
    x_val_np = _to_numpy_float(preprocessor.transform(x_val))
    x_test_np = _to_numpy_float(preprocessor.transform(x_test))
    y_fit_np = y_fit.to_numpy(dtype=np.float32)
    y_val_np = y_val.to_numpy(dtype=np.float32)
    y_test_np = y_test.to_numpy(dtype=np.float32)

    cv_metrics = run_cv(
        features=x_train,
        target=y_train,
        preprocessor=preprocessor,
        cv_folds=cv_folds,
        random_state=random_state,
        hidden_dims=hidden_dims,
        dropout=dropout,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        epochs=epochs,
        patience=patience,
        device_name=device_name,
    )

    _, best_epoch, best_val_loss, training_time_sec, _, model = train_one_split(
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
        random_state=random_state,
        device_name=device_name,
    )
    predictions, inference_time_sec = predict_with_model(model, x_test_np, device_name)

    results_df = pd.DataFrame([
        metric_row(
            predictions,
            y_test_np,
            training_time_sec=training_time_sec,
            inference_time_sec=inference_time_sec,
            train_rows=len(x_train),
            test_rows=len(x_test),
            raw_feature_count=x_train.shape[1],
            encoded_feature_count=x_fit_np.shape[1],
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            cv_folds=cv_folds,
            cv_metrics=cv_metrics,
            dataset_name=DATASET_NAME,
            target_column=TARGET_COLUMN,
        )
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    append_to_existing = output_path.exists() and output_path.stat().st_size > 0
    results_df.to_csv(
        output_path,
        mode="a" if append_to_existing else "w",
        header=not append_to_existing,
        index=False,
    )
    return results_df


def parse_hidden_dims(value: str) -> list[int]:
    dims = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not dims:
        raise argparse.ArgumentTypeError("At least one hidden layer size is required.")
    return dims


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a PyTorch MLP on BNN-UPC log_avg_delay."
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument(
        "--hidden-dims",
        type=parse_hidden_dims,
        default=parse_hidden_dims("128,64,32"),
        help="Comma-separated hidden layer widths, e.g. 128,64,32.",
    )
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument(
        "--device",
        default="auto",
        help="Use 'auto', 'cpu', or a PyTorch device string such as 'cuda'.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = train_bnnupc_mlp(
        input_path=args.input_path,
        output_path=args.output_path,
        test_size=args.test_size,
        random_state=args.random_state,
        cv_folds=args.cv_folds,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        device_name=args.device,
    )
    print(f"MLP results written to: {args.output_path}")
    print(results.to_string(index=False))
