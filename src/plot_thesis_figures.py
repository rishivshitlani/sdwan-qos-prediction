"""Generate result figures for the thesis from the committed evaluation CSVs.

All figures are produced from the reproducible per-class slice, SLA, and Bronze
threshold-sweep reports, so they stay consistent with the tables in the thesis.
Figures are written as vector PDFs into the thesis Figures/ directory and kept
deliberately plain (no frills) to match the project guidelines.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "reports" / "model_results"
FIGDIR = PROJECT_ROOT / "MSc AI-DA-AI Online Thesis Document" / "Figures"

MODEL_ORDER = ["XGBRegressor", "PyTorchMLP", "FTTransformer"]
MODEL_LABELS = {"XGBRegressor": "XGBoost", "PyTorchMLP": "MLP", "FTTransformer": "FT-Transformer"}
CLASS_ORDER = ["Gold", "Silver", "Bronze"]
# Muted, print-friendly palette (also distinguishable in greyscale).
MODEL_COLORS = {"XGBRegressor": "#4C72B0", "PyTorchMLP": "#DD8452", "FTTransformer": "#55A868"}
TUNED_SLA = {"Gold": 30.0, "Silver": 50.0, "Bronze": 60.0}


def _style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)


def _save(fig, name: str) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = FIGDIR / name
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(PROJECT_ROOT)}")


def _grouped_bar(values: dict[str, list[float]], *, ylabel: str, title: str, name: str,
                 ylim: tuple[float, float] | None = None, fmt: str = "{:.3f}") -> None:
    import numpy as np

    x = np.arange(len(CLASS_ORDER))
    width = 0.26
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for i, model in enumerate(MODEL_ORDER):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, values[model], width,
                      label=MODEL_LABELS[model], color=MODEL_COLORS[model])
        for b in bars:
            ax.annotate(fmt.format(b.get_height()),
                        (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_ORDER)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.28))
    _style(ax)
    _save(fig, name)


def per_class_delay_r2() -> None:
    df = pd.read_csv(RESULTS / "bnnupc_qos_slice_evaluation.csv")
    df = df[df.slice_type.eq("qos_class")].drop_duplicates(["model", "slice_value"], keep="last")
    values = {m: [float(df[(df.model == m) & (df.slice_value == c)]["r2_log_delay"].iloc[0])
                  for c in CLASS_ORDER] for m in MODEL_ORDER}
    _grouped_bar(values, ylabel=r"$R^2$ (log delay)",
                 title="Per-class delay prediction $R^2$ by model",
                 name="fig_perclass_r2.pdf", ylim=(0.0, 1.0))


def per_class_sla_f1() -> None:
    df = pd.read_csv(RESULTS / "bnnupc_sla_violation_precision.csv")
    # Keep only rows at the tuned thresholds, then latest run per (model, class).
    mask = df.apply(lambda r: TUNED_SLA.get(r["qos_class"]) == r["sla_threshold_ms"], axis=1)
    df = df[mask].drop_duplicates(["model", "qos_class"], keep="last")
    values = {m: [float(df[(df.model == m) & (df.qos_class == c)]["f1_score"].iloc[0])
                  for c in CLASS_ORDER] for m in MODEL_ORDER}
    _grouped_bar(values, ylabel="SLA F1 score",
                 title="Per-class SLA-violation F1 by model (tuned thresholds)",
                 name="fig_perclass_sla_f1.pdf", ylim=(0.0, 1.0))


def bronze_threshold_sweep() -> None:
    df = pd.read_csv(RESULTS / "bnnupc_bronze_threshold_sweep.csv")
    df = df.drop_duplicates("sla_threshold_ms", keep="last").sort_values("sla_threshold_ms")
    x = df["sla_threshold_ms"].to_numpy()
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for col, label, marker in [("precision", "Precision", "o"),
                               ("recall", "Recall", "s"),
                               ("f1_score", "F1", "^")]:
        ax.plot(x, df[col], marker=marker, label=label)
    ax.axvline(60, color="grey", linestyle=":", linewidth=1)
    ax.annotate("chosen (60 ms)", (60, 0.40), fontsize=8, rotation=90, va="bottom", ha="right", color="grey")
    ax.set_xlabel("Bronze SLA threshold (ms)")
    ax.set_ylabel("Score")
    ax.set_title("Bronze SLA threshold sensitivity (XGBoost)", fontsize=11)
    ax.invert_xaxis()  # show 100 -> 50 left-to-right (decreasing threshold)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    _save(fig, "fig_bronze_threshold_sweep.pdf")


def policy_r2() -> None:
    df = pd.read_csv(RESULTS / "bnnupc_qos_slice_evaluation.csv")
    df = df[(df.slice_type.eq("scheduling_policy")) & (df.model.eq("XGBRegressor"))]
    df = df.drop_duplicates("slice_value", keep="last")
    order = ["SP", "DRR", "WFQ"]
    vals = [float(df[df.slice_value == p]["r2_log_delay"].iloc[0]) for p in order]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    bars = ax.bar(order, vals, color="#4C72B0", width=0.55)
    for b in bars:
        ax.annotate(f"{b.get_height():.3f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=8)
    ax.set_ylabel(r"$R^2$ (log delay)")
    ax.set_title("Delay prediction $R^2$ by scheduling policy (XGBoost)", fontsize=11)
    ax.set_ylim(0.0, 1.0)
    _style(ax)
    _save(fig, "fig_policy_r2.pdf")


if __name__ == "__main__":
    print("Generating thesis figures...")
    per_class_delay_r2()
    per_class_sla_f1()
    bronze_threshold_sweep()
    policy_r2()
    print("Done.")
