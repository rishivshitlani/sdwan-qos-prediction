"""FT-Transformer (Feature Tokenizer + Transformer) for BNN-UPC delay.

This is a from-scratch PyTorch implementation of the FT-Transformer architecture
(Gorishniy et al., 2021, "Revisiting Deep Learning Models for Tabular Data"). It
is the modern, attention-based counterpart to the plain feedforward MLP: every
feature (numeric or categorical) is turned into a learned token, a [CLS] token is
prepended, a stack of Transformer blocks applies self-attention across the
feature tokens, and the final [CLS] representation is mapped to the prediction.

It uses the same raw feature set and the same log_avg_delay target as the MLP and
XGBoost baselines, so it can be dropped into the per-class QoS comparison.

Numeric features are median-imputed and standardised; categorical features are
integer-encoded (with a reserved index 0 for unseen/missing categories) and given
their own embedding tables inside the feature tokenizer.
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

from sdwan_qos.config import (
    BNNUPC_LEAKAGE_COLUMNS,
    BNNUPC_PROCESSED_DATASET,
    REPORTS_DIR,
)


DEFAULT_INPUT_PATH = BNNUPC_PROCESSED_DATASET
DEFAULT_OUTPUT_PATH = REPORTS_DIR / "bnnupc_ft_transformer_log_delay_results.csv"
TARGET_COLUMN = "log_avg_delay"
DATASET_NAME = "bnnupc_qos"

# Feature typing for the tokenizer. Small-cardinality codes (tos, time_distribution)
# and labels (qos_class, scheduling_policy) get embeddings; the rest are numeric.
CATEGORICAL_FEATURES = ["qos_class", "scheduling_policy", "time_distribution", "tos"]
NUMERIC_FEATURES = [
    "offered_bandwidth", "routing_hops", "n_nodes", "max_avg_lambda",
    "link_bandwidth", "tos_queue_weight", "min_tos_weight",
]


# ---------------------------------------------------------------------------
# Data loading and preprocessing
# ---------------------------------------------------------------------------

def load_features_and_target(input_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset does not exist: {input_path}")
    data = pd.read_csv(input_path)
    data.columns = data.columns.astype(str).str.strip()
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in {input_path}")
    drop = [c for c in [TARGET_COLUMN, *BNNUPC_LEAKAGE_COLUMNS] if c in data.columns]
    features = data.drop(columns=drop)
    target = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")
    mask = target.notna()
    return features.loc[mask].copy(), target.loc[mask].copy()


def _present(features: pd.DataFrame, columns: list[str]) -> list[str]:
    return [c for c in columns if c in features.columns]


def fit_preprocessor(train_features: pd.DataFrame) -> dict:
    """Fit numeric standardisation stats and categorical index maps on train only."""
    num_cols = _present(train_features, NUMERIC_FEATURES)
    cat_cols = _present(train_features, CATEGORICAL_FEATURES)

    num = train_features[num_cols].apply(pd.to_numeric, errors="coerce")
    medians = num.median()
    num = num.fillna(medians)
    means = num.mean()
    stds = num.std().replace(0.0, 1.0)  # guard constant columns (e.g. link_bandwidth)

    cat_maps: dict[str, dict[str, int]] = {}
    for c in cat_cols:
        values = sorted(train_features[c].astype(str).fillna("__nan__").unique())
        # index 0 is reserved for unseen / missing categories
        cat_maps[c] = {v: i + 1 for i, v in enumerate(values)}

    return {
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "medians": medians,
        "means": means,
        "stds": stds,
        "cat_maps": cat_maps,
    }


def cat_cardinalities(state: dict) -> list[int]:
    # +1 for the reserved unseen/missing index 0
    return [len(state["cat_maps"][c]) + 1 for c in state["cat_cols"]]


def transform(state: dict, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    num_cols, cat_cols = state["num_cols"], state["cat_cols"]

    num = features[num_cols].apply(pd.to_numeric, errors="coerce").fillna(state["medians"])
    x_num = ((num - state["means"]) / state["stds"]).to_numpy(dtype=np.float32)

    x_cat = np.zeros((len(features), len(cat_cols)), dtype=np.int64)
    for j, c in enumerate(cat_cols):
        mapping = state["cat_maps"][c]
        col = features[c].astype(str).fillna("__nan__")
        x_cat[:, j] = col.map(lambda v: mapping.get(v, 0)).to_numpy(dtype=np.int64)

    return x_num, x_cat


# ---------------------------------------------------------------------------
# FT-Transformer model
# ---------------------------------------------------------------------------

class FeatureTokenizer(nn.Module):
    """Turn numeric values and categorical indices into d-dimensional tokens."""

    def __init__(self, n_num: int, cat_cards: list[int], d_token: int):
        super().__init__()
        self.n_num = n_num
        if n_num > 0:
            self.num_weight = nn.Parameter(torch.empty(n_num, d_token))
            self.num_bias = nn.Parameter(torch.empty(n_num, d_token))
            nn.init.normal_(self.num_weight, std=0.02)
            nn.init.normal_(self.num_bias, std=0.02)
        self.cat_embeddings = nn.ModuleList([nn.Embedding(card, d_token) for card in cat_cards])
        for emb in self.cat_embeddings:
            nn.init.normal_(emb.weight, std=0.02)
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_token))
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        tokens = []
        if self.n_num > 0:
            # (b, n_num, d) = x[:, :, None] * W[None] + b[None]
            num_tokens = x_num.unsqueeze(-1) * self.num_weight.unsqueeze(0) + self.num_bias.unsqueeze(0)
            tokens.append(num_tokens)
        if len(self.cat_embeddings) > 0:
            cat_tokens = torch.stack(
                [emb(x_cat[:, j]) for j, emb in enumerate(self.cat_embeddings)], dim=1
            )
            tokens.append(cat_tokens)
        x = torch.cat(tokens, dim=1)  # (b, n_features, d)
        cls = self.cls_token.expand(x.shape[0], -1, -1)  # (b, 1, d)
        return torch.cat([cls, x], dim=1)  # (b, 1 + n_features, d)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: MHSA + FFN with residual connections."""

    def __init__(self, d_token: int, n_heads: int, ffn_hidden: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_token)
        self.attn = nn.MultiheadAttention(d_token, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_token)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, d_token),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.dropout(attn_out)
        h = self.norm2(x)
        x = x + self.dropout(self.ffn(h))
        return x


class FTTransformer(nn.Module):
    def __init__(
        self,
        n_num: int,
        cat_cards: list[int],
        *,
        d_token: int = 64,
        n_blocks: int = 3,
        n_heads: int = 8,
        ffn_factor: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_num, cat_cards, d_token)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_token, n_heads, d_token * ffn_factor, dropout)
            for _ in range(n_blocks)
        ])
        self.head_norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, 1)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        x = self.tokenizer(x_num, x_cat)
        for block in self.blocks:
            x = block(x)
        cls = x[:, 0]  # [CLS] token representation
        return self.head(F.relu(self.head_norm(cls)))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device_name: str) -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_split(
    x_num_train: np.ndarray,
    x_cat_train: np.ndarray,
    y_train: np.ndarray,
    x_num_eval: np.ndarray,
    x_cat_eval: np.ndarray,
    y_eval: np.ndarray,
    *,
    cat_cards: list[int],
    d_token: int,
    n_blocks: int,
    n_heads: int,
    ffn_factor: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    patience: int,
    random_state: int,
    device_name: str,
) -> tuple[np.ndarray, int, float, float, object]:
    set_reproducible_seed(random_state)
    device = _resolve_device(device_name)

    model = FTTransformer(
        x_num_train.shape[1], cat_cards,
        d_token=d_token, n_blocks=n_blocks, n_heads=n_heads,
        ffn_factor=ffn_factor, dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    train_ds = TensorDataset(
        torch.from_numpy(x_num_train),
        torch.from_numpy(x_cat_train),
        torch.from_numpy(y_train.reshape(-1, 1).astype(np.float32)),
    )
    generator = torch.Generator()
    generator.manual_seed(random_state)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator)

    x_num_eval_t = torch.from_numpy(x_num_eval).to(device)
    x_cat_eval_t = torch.from_numpy(x_cat_eval).to(device)
    y_eval_t = torch.from_numpy(y_eval.reshape(-1, 1).astype(np.float32)).to(device)

    best_state, best_eval_loss, best_epoch, no_improve = None, float("inf"), 0, 0
    t0 = perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        for bn, bc, by in train_loader:
            bn, bc, by = bn.to(device), bc.to(device), by.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(bn, bc), by)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            eval_loss = float(loss_fn(model(x_num_eval_t, x_cat_eval_t), y_eval_t).item())

        if eval_loss < best_eval_loss - 1e-6:
            best_eval_loss, best_epoch = eval_loss, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    training_time_sec = perf_counter() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        predictions = model(x_num_eval_t, x_cat_eval_t).detach().cpu().numpy().reshape(-1)

    return predictions, best_epoch, best_eval_loss, training_time_sec, model


def predict_with_model(model, x_num: np.ndarray, x_cat: np.ndarray, device_name: str) -> np.ndarray:
    device = _resolve_device(device_name)
    model.eval()
    with torch.no_grad():
        preds = model(
            torch.from_numpy(x_num).to(device),
            torch.from_numpy(x_cat).to(device),
        ).detach().cpu().numpy().reshape(-1)
    return preds


# ---------------------------------------------------------------------------
# Cross-validated global training entry point (mirrors train_bnnupc_mlp.py)
# ---------------------------------------------------------------------------

def run_cv(features: pd.DataFrame, target: pd.Series, *, cv_folds: int, random_state: int, hp: dict) -> dict:
    if cv_folds <= 1:
        return {"cv_folds": cv_folds, "cv_mae_mean": None, "cv_rmse_mean": None, "cv_r2_mean": None}
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    maes, rmses, r2s = [], [], []
    for fold_idx, (tr, va) in enumerate(cv.split(features), start=1):
        x_tr, x_va = features.iloc[tr], features.iloc[va]
        y_tr = target.iloc[tr].to_numpy(dtype=np.float32)
        y_va = target.iloc[va].to_numpy(dtype=np.float32)
        state = fit_preprocessor(x_tr)
        xn_tr, xc_tr = transform(state, x_tr)
        xn_va, xc_va = transform(state, x_va)
        preds, *_ = train_one_split(
            xn_tr, xc_tr, y_tr, xn_va, xc_va, y_va,
            cat_cards=cat_cardinalities(state), random_state=random_state + fold_idx,
            **hp,
        )
        maes.append(mean_absolute_error(y_va, preds))
        rmses.append(float(np.sqrt(mean_squared_error(y_va, preds))))
        r2s.append(r2_score(y_va, preds))
    return {
        "cv_folds": cv_folds,
        "cv_mae_mean": float(np.mean(maes)), "cv_mae_std": float(np.std(maes)),
        "cv_rmse_mean": float(np.mean(rmses)), "cv_rmse_std": float(np.std(rmses)),
        "cv_r2_mean": float(np.mean(r2s)), "cv_r2_std": float(np.std(r2s)),
    }


def train_bnnupc_ft_transformer(
    input_path: Path, output_path: Path, *, test_size: float, random_state: int,
    cv_folds: int, hp: dict, device_name: str,
) -> pd.DataFrame:
    features, target = load_features_and_target(input_path)
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state)
    x_fit, x_val, y_fit, y_val = train_test_split(
        x_train, y_train, test_size=0.10, random_state=random_state)

    cv_metrics = run_cv(x_train, y_train, cv_folds=cv_folds, random_state=random_state, hp=hp)

    state = fit_preprocessor(x_fit)
    xn_fit, xc_fit = transform(state, x_fit)
    xn_val, xc_val = transform(state, x_val)
    xn_test, xc_test = transform(state, x_test)

    _, best_epoch, best_val_loss, training_time_sec, model = train_one_split(
        xn_fit, xc_fit, y_fit.to_numpy(dtype=np.float32),
        xn_val, xc_val, y_val.to_numpy(dtype=np.float32),
        cat_cards=cat_cardinalities(state), random_state=random_state, **hp,
    )
    preds = predict_with_model(model, xn_test, xc_test, device_name)
    y_test_np = y_test.to_numpy(dtype=np.float32)

    row = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": DATASET_NAME, "model": "FTTransformer", "target": TARGET_COLUMN,
        "status": "ok", "error_message": "",
        "train_rows": len(x_train), "test_rows": len(x_test),
        "feature_count": features.shape[1],
        "mae": mean_absolute_error(y_test_np, preds),
        "rmse": float(np.sqrt(mean_squared_error(y_test_np, preds))),
        "r2_score": r2_score(y_test_np, preds),
        **cv_metrics,
        "best_epoch": best_epoch, "best_val_loss": best_val_loss,
        "training_time_sec": training_time_sec,
    }
    results_df = pd.DataFrame([row])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    append = output_path.exists() and output_path.stat().st_size > 0
    results_df.to_csv(output_path, mode="a" if append else "w", header=not append, index=False)
    return results_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an FT-Transformer on BNN-UPC log_avg_delay.")
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=5)
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


def hp_from_args(args: argparse.Namespace) -> dict:
    return {
        "d_token": args.d_token, "n_blocks": args.n_blocks, "n_heads": args.n_heads,
        "ffn_factor": args.ffn_factor, "dropout": args.dropout,
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "batch_size": args.batch_size, "epochs": args.epochs, "patience": args.patience,
        "device_name": args.device,
    }


if __name__ == "__main__":
    args = parse_args()
    results = train_bnnupc_ft_transformer(
        input_path=args.input_path, output_path=args.output_path,
        test_size=args.test_size, random_state=args.random_state,
        cv_folds=args.cv_folds, hp=hp_from_args(args), device_name=args.device,
    )
    print(f"FT-Transformer results written to: {args.output_path}")
    print(results.to_string(index=False))
