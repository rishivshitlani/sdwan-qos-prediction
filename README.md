# AI-Driven QoS Prediction in SD-WAN Networks

This repository contains the data pipelines, experiments, results, thesis material, and conference paper for an MSc Artificial Intelligence project on quality-of-service (QoS) prediction in software-defined wide area networks (SD-WANs).

The completed main study uses packet-level data generated with **BNNetSimulator**. It predicts per-flow delay for Gold, Silver, and Bronze traffic and evaluates whether those predictions can support SLA-risk detection and candidate weighted-fair-queuing (WFQ) allocation decisions.

## Project Status

The repository currently includes:

- generation and processing of a local BNN-UPC-style simulation dataset;
- leakage-safe classical regression baselines;
- row-wise XGBoost, multilayer perceptron (MLP), and FT-Transformer comparisons;
- QoS-class, scenario, scheduling-policy, tail-delay, jitter, and SLA analyses;
- model-significance and feature-importance utilities;
- an exploratory XGBoost evaluation grouped by simulation run (see [Grouped Evaluation](#grouped-evaluation-not-used-in-the-thesis-or-conference-paper); not used in the thesis or conference paper);
- an experimental WFQ allocation recommender;
- earlier Zenodo, CICIDS2017, and synthetic-data pipelines; and
- the MSc thesis and an IEEE-style conference paper.

The primary prediction target is `log_avg_delay`, the natural logarithm of average flow delay in seconds. Average delay in milliseconds is recovered with `exp(log_avg_delay) * 1000`. The reported QoS-aware model comparisons use shuffled row-wise five-fold cross-validation with a fixed random seed.

## Main Dataset and Results

The processed BNN-UPC dataset contains **29,280 flows from 400 simulation runs**. Each row represents a source-destination flow.

| QoS class | Flows | Row-wise log-delay R² | Delay MAE | SLA F1 |
| --- | ---: | ---: | ---: | ---: |
| Gold | 4,592 | 0.914 | 3.05 ms | 0.944 |
| Silver | 8,991 | 0.908 | 3.48 ms | 0.720 |
| Bronze | 15,697 | 0.704 | 20.68 ms | 0.592 |
| Overall | 29,280 | 0.766 | 12.63 ms | — |

These values are the checked-in row-wise XGBoost results in [`bnnupc_qos_slice_evaluation.csv`](reports/model_results/bnnupc_qos_slice_evaluation.csv) and [`bnnupc_sla_violation_precision.csv`](reports/model_results/bnnupc_sla_violation_precision.csv). `simulation_id` and `scenario` are excluded from the model features.

Additional row-wise findings are:

- 90th-percentile delay: overall R² = 0.287 and MAE = 23.77 ms;
- jitter: overall R² = 0.020 and MAE = 1.74 ms; and
- Gold SLA alerts: precision = 0.994, recall = 0.899, and F1 = 0.944.

The MLP reaches overall R² = 0.766 and the FT-Transformer reaches 0.762 under their corresponding row-wise outer-fold evaluations. These protocols evaluate held-out flow rows; they do not hold out complete simulation runs.

The complete interpretation and limitations are discussed in the [conference paper](IEEE-paper/IEEE_paper_main.pdf).

## Grouped Evaluation (Not Used in the Thesis or Conference Paper)

The row-wise cross-validation above splits individual flow rows, so rows from the same simulation run can land in different folds even though `simulation_id` is not used as a feature. To check whether that affects the reported scores, a separate XGBoost evaluation was run using `sklearn.model_selection.GroupKFold` with `simulation_id` as the grouping key, so every flow from a given simulation run stays entirely on one side of each fold.

This evaluation was completed and is fully reproducible: running [`src/evaluate_bnnupc_grouped_cv.py`](src/evaluate_bnnupc_grouped_cv.py) regenerates [`bnnupc_grouped_cv_xgb.csv`](reports/model_results/bnnupc_grouped_cv_xgb.csv), and [`src/evaluate_bnnupc_grouped_metrics.py`](src/evaluate_bnnupc_grouped_metrics.py) regenerates [`bnnupc_grouped_cv_metrics.csv`](reports/model_results/bnnupc_grouped_cv_metrics.csv).

| QoS class | Flows | Grouped log-delay R² | Delay MAE | SLA F1 |
| --- | ---: | ---: | ---: | ---: |
| Gold | 4,592 | 0.907 | 3.17 ms | 0.940 |
| Silver | 8,991 | 0.898 | 3.67 ms | 0.680 |
| Bronze | 15,697 | 0.680 | 21.57 ms | 0.554 |
| Overall | 29,280 | 0.747 | 13.19 ms | — |

Additional grouped findings: 90th-percentile delay overall R² = 0.293 (MAE = 32.51 ms); jitter overall R² = -0.040 (MAE = 3.22 ms); Gold SLA precision = 0.986, recall = 0.898, F1 = 0.940.

**This evaluation is not discussed in the MSc thesis and is not used in the conference paper.** Both of those documents report only the row-wise results in the section above. The grouped result is retained here as a supporting sensitivity check on the evaluation protocol, not as a competing headline claim.

## Repository Layout

```text
sdwan-qos-prediction/
├── IEEE-paper/             IEEE-style paper source and figures
├── data/
│   ├── raw/                 source datasets and simulator files
│   ├── processed/           processed modelling datasets and audit reports
│   └── synthetic/           generated development dataset
├── documents/               project notes and supporting documentation
├── MSc AI-DA-AI Online Thesis Document/
│                            thesis source, bibliography, figures, and PDF
├── output/pdf/              stable paper output
├── reports/model_results/   curated experiment results
├── src/                     data, modelling, evaluation, and plotting scripts
├── requirements.txt
└── README.md
```

Large raw datasets, the virtual environment, simulator output, and most generated model artifacts are excluded from Git. Curated CSV results needed to support the written analysis are retained in `reports/model_results/`.

## Setup

Python 3.10 or newer is recommended. Use the project virtual environment for all commands.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Check that every Python script parses correctly:

```bash
.venv/bin/python -m py_compile src/*.py
```

Docker Desktop is also required to generate new BNNetSimulator results.

## Reproduce the Main BNN-UPC Workflow

### 1. Generate simulator inputs

```bash
.venv/bin/python src/generate_bnnupc_dataset.py \
  --n-topologies 20 \
  --n-tms-per-topology 20 \
  --net-size-min 6 \
  --net-size-max 12
```

This creates graphs, routing matrices, traffic matrices, `simulation.txt`, and `conf.yml` under `data/raw/BNN_UPC/sim_input/`. With these arguments it creates 400 runs. Generation is deterministic for a fixed random seed.

Four experiment scenarios are represented:

| Scenario | Scheduling setup |
| --- | --- |
| A | WFQ with fixed Gold/Silver/Bronze weights of 60/30/10 |
| B | WFQ with varied queue-weight profiles |
| C | Mixed SP, WFQ, and DRR policies with varied profiles |
| D | Mixed policies with equally probable traffic-class assignment |

The project maps simulator ToS values to classes as follows:

| ToS | QoS class | Intended priority |
| ---: | --- | --- |
| 0 | Gold | Highest |
| 1 | Silver | Medium |
| 2 | Bronze | Lowest |

### 2. Run BNNetSimulator

From the repository root:

```bash
docker run --rm \
  --mount type=bind,src=$(pwd)/data/raw/BNN_UPC/sim_input,dst=/data \
  bnnupc/bnnetsimulator
```

The container writes compressed per-flow results beneath `data/raw/BNN_UPC/sim_input/results/qos_sdwan/`.

### 3. Build the modelling table

```bash
.venv/bin/python src/process_bnnupc_dataset.py
```

Output: `data/processed/bnnupc_qos_dataset.csv`

The table contains simulation context, traffic class, offered bandwidth, traffic distribution, route length, topology size, link properties, scheduler information, queue weights, delay percentiles, jitter, packet loss, achieved bandwidth, and the log-delay target.

### 4. Run the QoS-aware row-wise evaluation

```bash
.venv/bin/python src/evaluate_bnnupc_qos_slices.py
```

Outputs:

```text
reports/model_results/bnnupc_qos_slice_evaluation.csv
reports/model_results/bnnupc_sla_violation_precision.csv
```

This script creates out-of-fold XGBoost predictions using shuffled five-fold row-wise cross-validation, then reports overall, per-class, scenario, and scheduling-policy regression metrics plus SLA classification metrics.

### 5. Evaluate tail delay and jitter

```bash
.venv/bin/python src/evaluate_bnnupc_metric_slices.py
```

Output: `reports/model_results/bnnupc_metric_slice_evaluation.csv`

### 6. Generate analysis figures

```bash
.venv/bin/python src/plot_shap_importance.py
.venv/bin/python src/plot_thesis_figures.py
```

The figure scripts write into the thesis figure directory. SHAP values describe model associations and must not be interpreted as causal effects.

## Supporting Experiments

### Classical regression models

`train_baseline.py` supports a mean dummy regressor, linear regression, RBF SVR, random forest, and XGBoost. For the BNN-UPC log-delay task, use:

```bash
.venv/bin/python src/train_baseline.py \
  --input-path data/processed/bnnupc_qos_dataset.csv \
  --target-column log_avg_delay \
  --dataset-name bnnupc_qos \
  --output-path reports/model_results/bnnupc_log_delay_results.csv \
  --drop-column simulation_id \
  --drop-column scenario \
  --drop-column avg_delay \
  --drop-column jitter \
  --drop-column packet_loss_rate \
  --drop-column delay_p10 \
  --drop-column delay_p50 \
  --drop-column delay_p90 \
  --drop-column actual_bandwidth
```

The generic trainer records holdout and shuffled K-fold metrics and writes feature importance for compatible models. It does not group folds by simulation.

### Neural tabular models

```bash
.venv/bin/python src/train_bnnupc_mlp.py
.venv/bin/python src/train_bnnupc_ft_transformer.py
```

QoS-aware out-of-fold comparisons are produced with:

```bash
.venv/bin/python src/evaluate_bnnupc_mlp_slices.py
.venv/bin/python src/evaluate_bnnupc_ft_transformer_slices.py
```

### SLA threshold and packet-loss analysis

The QoS-aware evaluator reports row-wise out-of-fold results by QoS class, scenario, and scheduling policy. Default experimental SLA thresholds are 30 ms for Gold, 50 ms for Silver, and 60 ms for Bronze. They are research operating points, not contractual SLA values.

Reproduce the Bronze delay-threshold sensitivity analysis with:

```bash
.venv/bin/python src/evaluate_bnnupc_qos_slices.py --bronze-sweep
```

The Bronze threshold started at 100 ms. The sweep above was used to test lower values, and 60 ms was kept as the final threshold because it stays strictly above Silver's 50 ms threshold while increasing recall by about 54% relative to 100 ms (0.368 to 0.568), at a modest precision cost (0.726 to 0.619).

Evaluate Bronze packet-loss occurrence separately with:

```bash
.venv/bin/python src/evaluate_bnnupc_bronze_loss_classifier.py
```

Gold and Silver packet-loss events are too rare in the current simulations for equivalent class-specific classifiers.

### Model comparison utilities

```bash
.venv/bin/python src/compute_bnnupc_global_mae_ms.py
.venv/bin/python src/compute_model_significance.py
```

These create global millisecond comparisons and a paired XGBoost-versus-MLP significance report using the project's row-wise or row-split evaluation protocols.

## Experimental WFQ Allocation Recommender

```bash
.venv/bin/python src/recommend_qos_allocation.py
```

The recommender fits a delay model and scores candidate Gold/Silver/Bronze WFQ weight profiles using predicted delay, SLA feasibility, and weighted violation cost.

Output: `reports/model_results/bnnupc_qos_allocation_recommendations.csv`

This is a model-based what-if experiment. It is not a production SD-WAN controller, does not directly configure a network, and is not part of the conference paper's reported contributions.

## Earlier Dataset Pipelines

### Zenodo 13754300

The Zenodo 5G testbed dataset provides measured throughput, jitter, packet loss, and one-way delay. It is retained as the public-measurement baseline.

```bash
.venv/bin/python src/process_zenodo_dataset.py --skip-owd
.venv/bin/python src/train_zenodo_baseline.py --skip-owd
```

`--skip-owd` avoids aggregating the large packet-level one-way-delay files. The default modelling target is `actual_throughput_mbps`; the wrapper removes post-measurement fields that would leak the outcome. A derived `recommended_bandwidth_percent` target is also supported as a secondary compatibility experiment.

### CICIDS2017

CICIDS2017 is an intrusion-detection dataset, not a direct QoS dataset. It is used only for data-engineering exploration; any SD-WAN-aligned fields derived from it are proxies.

```bash
.venv/bin/python src/process_public_dataset.py
.venv/bin/python src/clean_public_dataset.py
.venv/bin/python src/create_cicids_project_aligned_sample.py
```

### Synthetic development data

```bash
.venv/bin/python src/generate_dataset.py
```

This creates `data/synthetic/sdwan_qos_synthetic.csv` with rule-based features and a derived bandwidth-recommendation target. It supports pipeline development and is not used for empirical performance claims.

## Leakage Controls

For the main BNN-UPC log-delay task:

- `simulation_id` and `scenario` are excluded from model inputs;
- average delay, delay percentiles, jitter, packet loss, and achieved bandwidth are excluded because they are outcomes of the same simulation;
- preprocessing is fitted independently within each fold; and
- evaluation rows are scored only from out-of-fold predictions.

The reported cross-validation splits individual flow rows. Consequently, flows from the same simulation run can appear in different folds even though `simulation_id` is not used as a feature.

For derived recommendation targets, observed outcomes used to calculate the target are removed from the feature matrix.

## Results and Documentation

- [`reports/model_results/README.md`](reports/model_results/README.md) maps curated CSV files to experiments.
- [`data/raw/BNN_UPC/README.md`](data/raw/BNN_UPC/README.md) documents local simulator-data generation.
- [`data/raw/Zenodo_13754300/README.md`](data/raw/Zenodo_13754300/README.md) describes the Zenodo source files.
- [`documents/qos_classes.md`](documents/qos_classes.md) defines the Gold, Silver, and Bronze policy.
- [`documents/model_explanation.md`](documents/model_explanation.md) explains the modelling pipeline.

Most result-producing scripts append timestamped rows when an output already exists, while some dedicated evaluators overwrite their CSVs. Check timestamps and evaluation labels before comparing runs.

## Limitations

- The main dataset is simulator-generated. The row-wise evaluation measures performance on held-out flow records from the same generator, not transfer to unseen simulation runs or a real enterprise network.
- The features are mainly static and do not include recent queue occupancy, rolling arrival rates, or live controller telemetry.
- Bronze delay is harder to predict than Gold or Silver delay; tail delay is only moderately predictable, and jitter is predicted weakly with the current features.
- SLA thresholds and candidate WFQ profiles are experimental choices.

## Author

Rishiv Shitlani — MSc Computer Science (Artificial Intelligence), University of Galway
