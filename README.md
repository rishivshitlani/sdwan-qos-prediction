# AI-Driven QoS Prediction in SD-WAN Networks

This repository contains the code, processed experiment data, results, thesis material, and conference paper for an MSc Artificial Intelligence project on quality-of-service (QoS) prediction in software-defined wide area networks (SD-WANs).

The current primary experiment uses packet-level simulations from **BNNetSimulator** to predict per-flow delay for Gold, Silver, and Bronze traffic. The evaluation is grouped by simulation run so that related flows from the same run cannot appear in both training and validation data.

## Research Focus

The project investigates three connected questions:

1. How accurately can machine-learning models predict continuous QoS outcomes for different traffic classes?
2. How does evaluation design, particularly row-wise versus simulation-grouped validation, affect the reported performance?
3. Can predicted QoS support SD-WAN decisions such as SLA-risk detection and candidate WFQ bandwidth allocation?

The main supervised-learning target is `log_avg_delay`, the natural logarithm of average per-flow delay. Secondary targets include 90th-percentile delay (`delay_p90`) and jitter.

## Current Headline Results

The primary dataset contains **29,280 flows from 400 simulation runs**:

| QoS class | Flows | Grouped log-delay R² | Delay MAE | SLA F1 |
| --- | ---: | ---: | ---: | ---: |
| Gold | 4,592 | 0.907 | 3.17 ms | 0.940 |
| Silver | 8,991 | 0.898 | 3.67 ms | 0.680 |
| Bronze | 15,697 | 0.680 | 21.57 ms | 0.554 |
| Overall | 29,280 | 0.747 | 13.19 ms | — |

These values come from five-fold `sklearn.model_selection.GroupKFold` evaluation using `simulation_id` as the grouping key. The identifier is used only to define folds and is excluded from model inputs.

A shuffled row-wise comparison produces a modestly higher overall R² of 0.766. The difference supports treating the simulation run, rather than an individual flow row, as the evaluation unit when flows from a run share topology, load, and scheduler context.

Under the same grouped protocol:

- 90th-percentile delay is moderately predictable overall (R² = 0.293).
- Jitter is not predicted reliably by the static feature set (R² = -0.040).
- Gold SLA alerts have 0.986 precision: 1,184 of 1,201 predicted violations are true violations.

The complete discussion and limitations are available in the [conference paper](output/pdf/SSGP26_SD-WAN_QoS_Paper_Draft.pdf). A [single-column reading version](output/pdf/SSGP26_SD-WAN_QoS_Paper_iPad_Reading.pdf) is also provided.

## Repository Structure

```text
sdwan-qos-prediction/
├── conference-paper/        IEEE-style paper source, figures, and build output
├── data/
│   ├── raw/                 source datasets and BNNetSimulator inputs/results
│   ├── processed/           project-aligned datasets
│   └── synthetic/           generated SD-WAN-style data
├── documents/               QoS-class and model documentation
├── MSc AI-DA-AI Online Thesis Document/
│                            thesis source, figures, bibliography, and PDF
├── output/pdf/              stable paper PDFs for reading and distribution
├── reports/model_results/   curated experiment metrics and recommendations
├── src/                     generation, processing, modelling, and evaluation
├── requirements.txt
└── README.md
```

Large raw datasets, the virtual environment, and most generated model artifacts are intentionally excluded from Git. Curated experiment CSVs under `reports/model_results/` are retained when they support the reported analysis.

## Setup

Python 3.10 or newer is recommended. Always run the project through its virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Docker Desktop is additionally required to generate packet-level results with BNNetSimulator.

To check the Python sources:

```bash
.venv/bin/python -m py_compile src/*.py
```

## Primary BNN-UPC Workflow

### 1. Generate BNNetSimulator inputs

```bash
.venv/bin/python src/generate_bnnupc_dataset.py \
  --n-topologies 20 \
  --n-tms-per-topology 20 \
  --net-size-min 6 \
  --net-size-max 12
```

This produces topology, routing, traffic-matrix, configuration, and manifest files under:

```text
data/raw/BNN_UPC/sim_input/
```

The default experiment creates 20 topologies and 20 traffic matrices per topology, giving 400 simulation runs. Generation is deterministic for a fixed random seed.

### 2. Run BNNetSimulator

```bash
docker run --rm \
  --mount type=bind,src=$(pwd)/data/raw/BNN_UPC/sim_input,dst=/data \
  bnnupc/bnnetsimulator
```

The official container writes compressed per-flow simulation results beneath `data/raw/BNN_UPC/sim_input/results/`.

### 3. Process the simulation results

```bash
.venv/bin/python src/process_bnnupc_dataset.py
```

Output:

```text
data/processed/bnnupc_qos_dataset.csv
```

Each row represents one source-destination flow. The processed features include traffic class, offered bandwidth, traffic distribution, route length, topology size, link properties, scheduling policy, and queue weights. Outcome columns include average delay, delay percentiles, jitter, packet loss, and achieved bandwidth.

BNNetSimulator ToS values are mapped as follows:

| ToS | QoS class | Priority |
| ---: | --- | --- |
| 0 | Gold | Highest |
| 1 | Silver | Medium |
| 2 | Bronze | Lowest |

### 4. Run the primary grouped evaluation

```bash
.venv/bin/python src/evaluate_bnnupc_grouped_cv.py
```

Output:

```text
reports/model_results/bnnupc_grouped_cv_xgb.csv
```

The evaluator trains XGBoost on `log_avg_delay`, creates out-of-fold predictions with `GroupKFold`, and reports overall and per-class regression and SLA-trigger metrics. The row-wise sensitivity results are produced separately by `evaluate_bnnupc_qos_slices.py`.

### 5. Evaluate tail delay and jitter with the same folds

```bash
.venv/bin/python src/evaluate_bnnupc_grouped_metrics.py
```

Output:

```text
reports/model_results/bnnupc_grouped_cv_metrics.csv
```

### 6. Generate SHAP and thesis figures

```bash
.venv/bin/python src/plot_shap_importance.py
.venv/bin/python src/plot_thesis_figures.py
```

These scripts write publication figures to the thesis figure directory. The SHAP plots describe model associations, not causal effects.

## Additional Model Comparisons

### Classical models

The generic trainer supports leakage-safe preprocessing for linear regression, SVR, random forest, and XGBoost:

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

### Neural tabular models

```bash
.venv/bin/python src/train_bnnupc_mlp.py
.venv/bin/python src/train_bnnupc_ft_transformer.py
```

Class- and policy-aware comparisons can be produced with:

```bash
.venv/bin/python src/evaluate_bnnupc_mlp_slices.py
.venv/bin/python src/evaluate_bnnupc_ft_transformer_slices.py
```

The existing neural comparisons use row-wise outer folds and are therefore supporting sensitivity results, not substitutes for the primary simulation-grouped estimate.

## SLA and QoS-Slice Evaluation

The general slice evaluator reports metrics by class, scenario, and scheduling policy:

```bash
.venv/bin/python src/evaluate_bnnupc_qos_slices.py
```

Default experimental SLA thresholds are:

```text
Gold:   30 ms
Silver: 50 ms
Bronze: 60 ms
```

These thresholds are research operating points, not contractual SLA values. The Bronze threshold can be examined with a sensitivity sweep:

```bash
.venv/bin/python src/evaluate_bnnupc_qos_slices.py --bronze-sweep
```

Bronze packet-loss risk can be evaluated separately:

```bash
.venv/bin/python src/evaluate_bnnupc_bronze_loss_classifier.py
```

Gold and Silver loss events are too rare in the current dataset for a meaningful equivalent classifier.

## Experimental Layer 3 Allocation Recommender

The project also contains a what-if recommender for candidate WFQ class weights:

```bash
.venv/bin/python src/recommend_qos_allocation.py
```

It scores candidate Gold/Silver/Bronze profiles by predicted class delay, SLA feasibility, and weighted violation cost. This is an experimental extension built on model predictions; it is not part of the conference paper's reported contributions and is not a production SD-WAN controller.

Output:

```text
reports/model_results/bnnupc_qos_allocation_recommendations.csv
```

## Other Dataset Paths

### Zenodo 13754300

The Zenodo 5G testbed dataset provides direct throughput, jitter, packet-loss, and one-way-delay measurements. It remains the main public-measurement baseline.

Fast processing without the large packet-level OWD aggregation:

```bash
.venv/bin/python src/process_zenodo_dataset.py --skip-owd
.venv/bin/python src/train_zenodo_baseline.py --skip-owd
```

The default Zenodo target is `actual_throughput_mbps`. Outcome fields such as derived bandwidth recommendations, jitter, and packet loss are excluded when they would leak post-measurement information into a deployment-style prediction task.

### CICIDS2017

CICIDS2017 is retained for data-engineering exploration only. It is an intrusion-detection dataset rather than a QoS dataset, so SD-WAN-aligned fields derived from it are proxies and are not used as primary QoS evidence.

```bash
.venv/bin/python src/process_public_dataset.py
.venv/bin/python src/clean_public_dataset.py
.venv/bin/python src/create_cicids_project_aligned_sample.py
```

### Synthetic data

```bash
.venv/bin/python src/generate_dataset.py
```

This creates `data/synthetic/sdwan_qos_synthetic.csv` with rule-based SD-WAN-style features and a derived bandwidth-recommendation target. It is intended for pipeline development, not empirical performance claims.

## Leakage Prevention

The project applies the following rules:

- `simulation_id` and `scenario` are excluded from BNN-UPC model inputs.
- Raw delay, jitter, packet loss, percentiles, and achieved bandwidth are excluded when predicting `log_avg_delay`.
- Preprocessing is fitted independently inside each training fold.
- The primary BNN-UPC evaluation groups all flows from the same simulation run into the same fold.
- When predicting a derived recommendation target, any observed outcome used to construct that target is excluded from the feature matrix.

## Results and Documentation

- `reports/model_results/README.md` maps each curated result CSV to its experiment.
- `data/raw/BNN_UPC/README.md` documents local BNNetSimulator generation.
- `data/raw/Zenodo_13754300/README.md` documents the Zenodo source files.
- `documents/qos_classes.md` defines the Gold, Silver, and Bronze policy.
- `documents/model_explanation.md` explains the main modelling pipeline.

## Limitations

- The primary dataset is simulator-generated; grouped validation tests unseen runs from the same generator, not transfer to a real enterprise network.
- The current feature set is mostly static and does not include recent queue occupancy, arrival-rate windows, or controller telemetry.
- Tail delay is substantially harder to predict than average delay, and jitter does not generalise reliably from the current features.
- Neural comparisons have not yet been repeated under the same simulation-grouped outer-fold protocol.
- SLA thresholds and WFQ allocation profiles are experimental operating points.

## Author

Rishiv Shitlani

MSc Computer Science (Artificial Intelligence)

University of Galway
