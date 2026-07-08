"""SHAP feature-importance figures for the BNN-UPC XGBoost delay model.

Produces two vector PDFs into the thesis Figures/ directory:
  fig_shap_summary.pdf  -- SHAP beeswarm summary (per-sample impact + direction)
  fig_shap_bar.pdf      -- mean |SHAP| bar chart (global ranking)

SHAP explains *how* each feature moves the log-delay prediction, which is more
informative than XGBoost's built-in gain importance. The model, features, and
log-delay target match the thesis exactly (leakage-safe drop columns, no scaling
for the tree model), so the explanation is consistent with the reported results.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

from train_baseline import build_model_specs, build_preprocessor
from evaluate_bnnupc_qos_slices import load_bnnupc_dataset, split_features_and_target

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "bnnupc_qos_dataset.csv"
FIGDIR = PROJECT_ROOT / "MSc AI-DA-AI Online Thesis Document" / "Figures"
RANDOM_STATE = 42

# Readable names for the encoded feature columns.
PRETTY = {
    "numeric__routing_hops": "routing hops",
    "numeric__offered_bandwidth": "offered bandwidth",
    "numeric__max_avg_lambda": "max avg lambda",
    "numeric__tos_queue_weight": "queue weight",
    "numeric__min_tos_weight": "min queue weight",
    "numeric__n_nodes": "n nodes",
    "numeric__time_distribution": "time distribution",
    "numeric__tos": "ToS",
    "numeric__link_bandwidth": "link bandwidth",
    "categorical__qos_class_Gold": "class=Gold",
    "categorical__qos_class_Silver": "class=Silver",
    "categorical__qos_class_Bronze": "class=Bronze",
    "categorical__scheduling_policy_SP": "policy=SP",
    "categorical__scheduling_policy_WFQ": "policy=WFQ",
    "categorical__scheduling_policy_DRR": "policy=DRR",
}


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    data = load_bnnupc_dataset(DEFAULT_INPUT_PATH)
    features, target = split_features_and_target(data)

    spec = {s.name: s for s in build_model_specs(RANDOM_STATE)}["XGBRegressor"]
    preprocessor = build_preprocessor(features, scale_numeric=spec.scale_numeric)
    x_encoded = preprocessor.fit_transform(features)
    feature_names = [PRETTY.get(n, n) for n in preprocessor.get_feature_names_out()]

    model = spec.model
    model.fit(x_encoded, target.to_numpy(dtype=float))

    # TreeExplainer is exact and fast for XGBoost.
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_encoded)

    # Beeswarm summary.
    plt.figure()
    shap.summary_plot(shap_values, x_encoded, feature_names=feature_names, show=False, max_display=12)
    plt.title("SHAP summary: impact of each feature on predicted log delay", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIGDIR / "fig_shap_summary.pdf", bbox_inches="tight")
    plt.close()
    print(f"  wrote {FIGDIR / 'fig_shap_summary.pdf'}")

    # Mean |SHAP| bar.
    plt.figure()
    shap.summary_plot(shap_values, x_encoded, feature_names=feature_names,
                      plot_type="bar", show=False, max_display=12)
    plt.title("Global feature importance (mean |SHAP|)", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIGDIR / "fig_shap_bar.pdf", bbox_inches="tight")
    plt.close()
    print(f"  wrote {FIGDIR / 'fig_shap_bar.pdf'}")

    # Print the ranking for the thesis text.
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    print("\nMean |SHAP| ranking:")
    for i in order[:8]:
        print(f"  {feature_names[i]:20s} {mean_abs[i]:.4f}")


if __name__ == "__main__":
    print("Generating SHAP figures for XGBoost delay model...")
    main()
    print("Done.")
