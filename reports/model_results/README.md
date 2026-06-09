# Model Result Reports

This directory contains curated CSV outputs from the project experiments. Scripts generally append timestamped rows to existing CSVs instead of overwriting them, so use the `run_timestamp`, `dataset`, `model`, and `target` columns to identify a specific run.

The top-level `reports/model_results/` directory is the canonical report location. Any similarly named reports under generated or copied folders should be treated as snapshots unless they are intentionally promoted here.

## Main Reports

| File | Purpose |
| --- | --- |
| `model_results.csv` | Early synthetic SD-WAN baseline for `recommended_bandwidth_percent`. |
| `zenodo_baseline_results.csv` | Main Zenodo baseline for `actual_throughput_mbps`. Use this for the primary public QoS throughput experiment. |
| `zenodo_baseline_results_feature_importance.csv` | Feature-importance rows for Zenodo tree-based models. |
| `zenodo_baseline_recommended_bandwidth_results.csv` | Secondary Zenodo experiment for derived `recommended_bandwidth_percent`. |
| `bnnupc_baseline_results.csv` | Earlier BNN-UPC baseline for raw `avg_delay`. Superseded for most discussion by the log-delay reports. |
| `bnnupc_baseline_results_feature_importance.csv` | Feature-importance rows for the earlier raw-delay BNN-UPC baseline. |
| `bnnupc_log_delay_results.csv` | Main BNN-UPC classical baselines for `log_avg_delay`. |
| `bnnupc_log_delay_results_feature_importance.csv` | Feature-importance rows for the BNN-UPC `log_avg_delay` task. |
| `bnnupc_mlp_log_delay_results.csv` | Standalone PyTorch MLP result for `log_avg_delay`. |
| `bnnupc_qos_slice_evaluation.csv` | QoS-aware out-of-fold delay evaluation by overall, QoS class, scenario, and scheduling policy. Contains the checked-in `FTTransformer` delay results. |
| `bnnupc_sla_violation_precision.csv` | SLA violation precision, recall, and F1 by QoS class. Contains the checked-in `FTTransformer` SLA-trigger results. |
| `bnnupc_metric_slice_evaluation.csv` | Additional BNN-UPC slice evaluation for `jitter` and `delay_p90`. |
| `bnnupc_qos_allocation_recommendations.csv` | Candidate Gold/Silver/Bronze bandwidth allocation rankings. |
| `bnnupc_bronze_loss_classifier.csv` | Bronze packet-loss risk classifier results. |
| `bnnupc_bronze_threshold_sweep.csv` | Bronze SLA-threshold sensitivity results. |

## FT-Transformer Location

The current checked-in FT-Transformer results are not in a standalone `bnnupc_ft_transformer_log_delay_results.csv` file. They are stored as `FTTransformer` rows in:

```text
bnnupc_qos_slice_evaluation.csv
bnnupc_sla_violation_precision.csv
```

The standalone file `bnnupc_ft_transformer_log_delay_results.csv` is only produced if `src/train_bnnupc_ft_transformer.py` is run and its output is saved.

Current checked-in FT-Transformer delay summary:

| Scope | MAE log delay | RMSE log delay | R2 log delay | MAE delay |
| --- | ---: | ---: | ---: | ---: |
| Overall | 0.1817 | 0.3591 | 0.7615 | 12.59 ms |
| Gold | 0.1044 | 0.1786 | 0.8996 | 3.20 ms |
| Silver | 0.1124 | 0.1842 | 0.8977 | 3.53 ms |
| Bronze | 0.2440 | 0.4602 | 0.7027 | 20.52 ms |

## Which Files To Use In The Thesis

Use `zenodo_baseline_results.csv` for the main public QoS throughput baseline.

Use `bnnupc_log_delay_results.csv`, `bnnupc_mlp_log_delay_results.csv`, `bnnupc_qos_slice_evaluation.csv`, and `bnnupc_sla_violation_precision.csv` for the topology-aware BNN-UPC modelling comparison.

Use `bnnupc_qos_allocation_recommendations.csv` when discussing Gold/Silver/Bronze bandwidth allocation decisions.
