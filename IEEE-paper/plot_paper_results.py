"""Generate compact validation-sensitivity figures for the conference paper."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["pdf.fonttype"] = 42


ROOT = Path(__file__).resolve().parents[1]
ROW_RESULTS = ROOT / "reports" / "model_results" / "bnnupc_qos_slice_evaluation.csv"
ROW_SLA = ROOT / "reports" / "model_results" / "bnnupc_sla_violation_precision.csv"
GROUP_RESULTS = ROOT / "reports" / "model_results" / "bnnupc_grouped_cv_xgb.csv"
OUTPUT = Path(__file__).resolve().parent / "figures" / "validation_sensitivity.pdf"
CLASSES = ["Gold", "Silver", "Bronze"]


def latest_rows(frame: pd.DataFrame, model: str) -> pd.DataFrame:
    subset = frame.loc[frame["model"].eq(model)].copy()
    timestamp = subset["run_timestamp"].max()
    return subset.loc[subset["run_timestamp"].eq(timestamp)].copy()


row_reg = latest_rows(pd.read_csv(ROW_RESULTS), "XGBRegressor")
row_sla = latest_rows(pd.read_csv(ROW_SLA), "XGBRegressor")
grouped = pd.read_csv(GROUP_RESULTS)

row_r2 = [
    float(
        row_reg.loc[
            row_reg["slice_type"].eq("qos_class")
            & row_reg["slice_value"].eq(qos),
            "r2_log_delay",
        ].iloc[0]
    )
    for qos in CLASSES
]
group_r2 = [
    float(
        grouped.loc[
            grouped["slice_type"].eq("qos_class")
            & grouped["slice_value"].eq(qos),
            "r2_log_delay",
        ].iloc[0]
    )
    for qos in CLASSES
]
row_f1 = [
    float(row_sla.loc[row_sla["qos_class"].eq(qos), "f1_score"].iloc[0])
    for qos in CLASSES
]
group_f1 = [
    float(
        grouped.loc[
            grouped["slice_type"].eq("sla")
            & grouped["slice_value"].eq(qos),
            "f1_score",
        ].iloc[0]
    )
    for qos in CLASSES
]

x = np.arange(len(CLASSES))
width = 0.36
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.55))

for ax, row_values, group_values, ylabel, title in [
    (axes[0], row_r2, group_r2, r"$R^2$ (log delay)", "Delay prediction"),
    (axes[1], row_f1, group_f1, "F1", "SLA-violation detection"),
]:
    first = ax.bar(x - width / 2, row_values, width, label="Row K-fold", color="#4C78A8")
    second = ax.bar(
        x + width / 2,
        group_values,
        width,
        label="Grouped by simulation",
        color="#F58518",
    )
    ax.set_xticks(x, CLASSES)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(first, fmt="%.3f", fontsize=7, padding=2)
    ax.bar_label(second, fmt="%.3f", fontsize=7, padding=2)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
fig.tight_layout(rect=(0, 0.13, 1, 1))
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT, bbox_inches="tight")
print(f"Wrote {OUTPUT}")
