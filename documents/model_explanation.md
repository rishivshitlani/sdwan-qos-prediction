# Model Explanation: SD-WAN QoS Prediction

This document explains the machine learning models used in this project, the input features fed into each model, the output target being predicted, and the evaluation metrics used to compare model performance.

---

## 1. What the Models Are Predicting

### Zenodo Output (Target Variable)

All models in the Zenodo experiment are trained to predict one value:

| Target column | Unit | Description |
| --- | --- | --- |
| `actual_throughput_mbps` | Mbps | Achieved throughput measured at the receiver during an iperf3 test |

This is a **supervised regression** task — the model is given a set of network and radio configuration inputs and must output a continuous numeric prediction of how much throughput will actually be delivered.

A secondary derived target (`recommended_bandwidth_percent`) is also available. It is computed as `actual_throughput_mbps / offered_throughput_mbps × 100`. It is not used as the primary target because it is derived from the actual throughput — training on it while including `actual_throughput_mbps` as an input would give the model the answer indirectly (target leakage).

### BNN-UPC Output (Target Variable)

The BNN-UPC experiment predicts per-flow delay from topology, traffic, and QoS scheduling features.

| Target column | Unit / scale | Description |
| --- | --- | --- |
| `avg_delay` | simulator delay unit | Mean per-flow delay from BNNetSimulator |
| `log_avg_delay` | log-transformed delay | Main BNN-UPC modelling target, used to reduce the effect of the heavy-tailed raw delay distribution |

The current BNN-UPC baseline and neural-network experiments use `log_avg_delay` as the target. Raw delay is still retained in the processed dataset for analysis, but it is dropped from model inputs when predicting `log_avg_delay`.

---

## 2. Input Features

After loading the processed Zenodo dataset, the following columns are used as model inputs. Leakage columns (`recommended_bandwidth_percent`, `jitter_ms`, `packet_loss_percent`, `actual_throughput_mbps`) and traceability columns (`source_file`) are excluded from training.

### Numeric Features (17 total after preprocessing)

| Feature | Source | Description |
| --- | --- | --- |
| `latency_ms` | OWD packet files | Mean one-way packet delay derived from inter-arrival times (IAT), aggregated per radio configuration |
| `jitter_ms` | Throughput files | Mean jitter reported by iperf3 — excluded when predicting throughput to avoid leakage |
| `packet_loss_percent` | Throughput files | Mean packet loss reported by iperf3 — excluded when predicting throughput |
| `bandwidth_utilization_percent` | Derived | Offered throughput as a percentage of assumed link capacity (150 Mbps) |
| `flow_duration_sec` | OWD packet files | Estimated flow duration derived from packet inter-arrival times |
| `packet_count` | OWD packet files | Number of packets observed in the OWD measurement |
| `time_of_day` | OWD packet files | Hour of day (0–23) when the measurement began |
| `bw_mhz` | Radio config | Configured 5G radio bandwidth in MHz (e.g., 20 MHz) |
| `slots` | Radio config | Number of configured downlink slots in the 5G frame |
| `ratio` | Radio config | Configured downlink-to-uplink slot ratio |
| `rep` | Experiment metadata | Repetition number for the measurement run |
| `offered_throughput_mbps` | Throughput files | Target throughput configured in the iperf3 test (offered load) |

### Categorical Features (encoded as one-hot)

| Feature | Values | Description |
| --- | --- | --- |
| `protocol` | UDP | Transport protocol used (constant in Zenodo — all tests use UDP) |
| `application_type` | throughput_test | Traffic type (constant in Zenodo) |
| `link_type` | 5G | Underlay link type (constant in Zenodo) |
| `direction` | uplink, downlink | Whether the flow is uplink or downlink |
| `gnb` | oai, srs | gNB (5G base station) software implementation used |
| `sdr` | b200 | Software-defined radio hardware used |
| `testbed` | ntnu, wue | Which of the two 5G testbeds the measurement came from |

> **Note:** `protocol`, `application_type`, `link_type`, and `sdr` are constant across all Zenodo rows. They carry zero variance and contribute nothing to predictions. Feature importance analysis confirms their importance scores are 0.0. They will be dropped in future experiments.

### BNN-UPC Features

The BNN-UPC processed dataset is generated from BNNetSimulator archives. Each row represents one non-self source-destination flow from one simulation sample.

The leakage-safe feature set used by XGBoost and the PyTorch MLP contains 11 raw columns:

| Feature | Type | Description |
| --- | --- | --- |
| `tos` | numeric | BNNetSimulator traffic class identifier: 0=Gold, 1=Silver, 2=Bronze |
| `qos_class` | categorical | Human-readable QoS class derived from `tos` |
| `offered_bandwidth` | numeric | Offered traffic load for the flow |
| `time_distribution` | numeric | Traffic generation distribution code from the traffic matrix |
| `routing_hops` | numeric | Number of hops on the source-destination route |
| `n_nodes` | numeric | Number of nodes in the topology |
| `max_avg_lambda` | numeric | Maximum traffic intensity used for the simulation sample |
| `link_bandwidth` | numeric | Link capacity used by the generated topology |
| `scheduling_policy` | categorical | Dominant scheduling policy on the flow path (`WFQ`, `DRR`, or `SP`) |
| `tos_queue_weight` | numeric | Average queue weight for the flow's ToS along the path |
| `min_tos_weight` | numeric | Minimum queue weight for the flow's ToS along the path |

The following columns are excluded from BNN-UPC model inputs:

```text
simulation_id, scenario,
avg_delay, jitter, packet_loss_rate,
delay_p10, delay_p50, delay_p90,
actual_bandwidth
```

The first two are identifiers or experiment labels. The remaining columns are measured outcomes, so using them as inputs would leak information about the target.

---

## 3. Models

### 3.1 DummyRegressor (Mean Strategy)

**What it does:**
The DummyRegressor does not learn from the input features at all. It simply calculates the mean of the training target values and predicts that same mean for every new input, regardless of what the features say.

**Why it is included:**
It acts as a lower-bound sanity check. Any real model that cannot outperform this trivial baseline is useless. If a model's R² is close to 0 or negative, it is performing no better than simply predicting the average.

**How it works:**
```
prediction = mean(y_train)   for every row, regardless of X
```

**Hyperparameters:** None (strategy = "mean")

**Preprocessing:** No scaling needed (features are ignored).

**Results on Zenodo dataset:**

| Metric | Value |
| --- | --- |
| MAE | 36.90 Mbps |
| RMSE | 43.88 Mbps |
| R² | -0.007 |
| CV R² (mean ± std) | -0.011 ± 0.006 |
| Inference time | 0.011 ms/row |

---

### 3.2 Linear Regression

**What it does:**
Linear Regression fits a straight-line relationship between each input feature and the target. It finds the set of weights (coefficients) for each feature that minimises the total squared prediction error across all training rows.

**How it works:**
```
prediction = w1×feature1 + w2×feature2 + ... + wn×featureN + bias
```
The model learns the weight for each feature during training. Features with larger absolute weights have more influence on the prediction.

**Why it is included:**
It is the simplest interpretable model that actually learns from the data. It serves as the first real baseline above the DummyRegressor. A high R² for Linear Regression suggests that the relationship between features and throughput is largely linear — which is plausible here because `offered_throughput_mbps` and actual throughput have a near-linear relationship under stable radio conditions.

**Hyperparameters:** None (standard Ordinary Least Squares).

**Preprocessing:** Numeric features are scaled with `StandardScaler` (zero mean, unit variance) because feature magnitude affects coefficient size in linear models.

**Results on Zenodo dataset:**

| Metric | Value |
| --- | --- |
| MAE | 10.55 Mbps |
| RMSE | 14.15 Mbps |
| R² | 0.895 |
| CV R² (mean ± std) | 0.851 ± 0.019 |
| Inference time | 0.011 ms/row |

---

### 3.3 Support Vector Regression (SVR)

**What it does:**
SVR is a kernel-based method that finds a function fitting within a tube of width ε (epsilon) around the target values. Predictions that fall inside the tube incur no penalty. Only points outside the tube (support vectors) influence the model boundary. The RBF (Radial Basis Function) kernel maps the data into a higher-dimensional space, allowing SVR to capture non-linear relationships without explicitly expanding the feature set.

**How it works:**
The RBF kernel computes similarity between two data points:
```
K(x, z) = exp(-γ × ||x - z||²)
```
Points closer together in feature space get higher kernel similarity scores, allowing the model to learn locally smooth non-linear patterns.

**Key hyperparameters used:**

| Parameter | Value | Effect |
| --- | --- | --- |
| `kernel` | rbf | Uses the Radial Basis Function for non-linear mapping |
| `C` | 10.0 | Regularisation — higher C allows more complex fits, lower C is smoother |
| `epsilon` | 0.1 | Width of the no-penalty tube around predictions (in target units, Mbps) |

**Preprocessing:** Numeric features are scaled with `StandardScaler`. SVR is sensitive to feature scale — unscaled inputs would cause features with large numeric ranges (e.g., `offered_throughput_mbps` in the hundreds) to dominate the kernel distance computation unfairly.

**Results on Zenodo dataset:**

| Metric | Value |
| --- | --- |
| MAE | 5.69 Mbps |
| RMSE | 8.76 Mbps |
| R² | 0.960 |
| CV R² (mean ± std) | 0.924 ± 0.028 |
| Inference time | 0.022 ms/row |

SVR sits cleanly between Linear Regression and the tree-based models, capturing non-linear patterns that Linear Regression misses without the variance sensitivity of Random Forest.

---

### 3.4 Random Forest Regressor

**What it does:**
Random Forest builds a large number of decision trees (200 in this project), each trained on a randomly sampled subset of the training rows (bootstrap sampling) and using a randomly chosen subset of features at each split. The final prediction is the average of all 200 trees. This averaging process reduces overfitting compared to a single deep decision tree.

**How it works:**
Each tree splits the data by asking binary questions about features:
```
"Is offered_throughput_mbps > 50?"  → Yes: go right, No: go left
```
Each leaf node predicts the average target value of the training rows that reached it. The forest averages 200 such predictions.

**Key hyperparameters used:**

| Parameter | Value | Effect |
| --- | --- | --- |
| `n_estimators` | 200 | Number of trees — more trees reduce variance but increase training time |
| `random_state` | 42 | Seed for reproducibility |
| `n_jobs` | -1 | Use all available CPU cores for parallel tree building |

**Preprocessing:** No scaling needed. Decision trees split on thresholds, not on feature magnitude, so scaling does not affect splits or results.

**Feature importances (top 5):**

| Rank | Feature | Importance |
| --- | --- | --- |
| 1 | bandwidth_utilization_percent | 25.6% |
| 2 | offered_throughput_mbps | 21.8% |
| 3 | direction (uplink) | 17.2% |
| 4 | direction (downlink) | 12.0% |
| 5 | latency_ms | 4.3% |

Random Forest uses Mean Decrease in Impurity (MDI) to measure importance: how much each feature reduces prediction error across all splits in all trees.

**Results on Zenodo dataset:**

| Metric | Value |
| --- | --- |
| MAE | 2.82 Mbps |
| RMSE | 6.21 Mbps |
| R² | 0.980 |
| CV R² (mean ± std) | 0.936 ± 0.044 |
| Training time | 0.083 sec |
| Inference time | 0.121 ms/row |

---

### 3.5 XGBoost (Extreme Gradient Boosting)

**What it does:**
XGBoost builds trees sequentially rather than in parallel like Random Forest. Each new tree is trained to correct the errors (residuals) left by all the trees built before it. This iterative error-correction process, called gradient boosting, makes XGBoost highly accurate on structured tabular data.

**How it works:**
```
Tree 1: predict from raw features → residual 1
Tree 2: predict residual 1        → residual 2
Tree 3: predict residual 2        → residual 3
...
Final prediction = sum of all tree outputs × learning_rate
```
The `learning_rate` (shrinkage) scales each tree's contribution, preventing any single tree from dominating and helping the model generalise.

**Key hyperparameters used:**

| Parameter | Value | Effect |
| --- | --- | --- |
| `n_estimators` | 200 | Number of boosting rounds (trees) |
| `learning_rate` | 0.05 | Step size — small value means more trees needed but better generalisation |
| `max_depth` | 4 | Maximum depth of each tree — limits complexity, reduces overfitting |
| `subsample` | 0.9 | Fraction of training rows sampled per tree — adds randomness to reduce overfitting |
| `colsample_bytree` | 0.9 | Fraction of features sampled per tree |
| `objective` | reg:squarederror | Loss function — minimises squared error, standard for regression |

**Preprocessing:** No scaling needed. Tree-based splits are scale-invariant.

**Feature importances (top 5):**

| Rank | Feature | Importance |
| --- | --- | --- |
| 1 | offered_throughput_mbps | 46.6% |
| 2 | bandwidth_utilization_percent | 38.9% |
| 3 | latency_ms | 3.7% |
| 4 | gnb (oai) | 3.6% |
| 5 | testbed (wue) | 1.7% |

XGBoost uses gain-based importance: how much each feature reduces the loss function when used in a split, weighted by the number of training rows in that split. This differs from Random Forest's MDI, which is why `direction` appears important in RF but 0 in XGBoost — `offered_throughput_mbps` already captures direction implicitly (UL and DL have different offered loads), so XGBoost does not need `direction` as a separate signal once it has selected `offered_throughput_mbps`.

**Results on Zenodo dataset:**

| Metric | Value |
| --- | --- |
| MAE | 3.29 Mbps |
| RMSE | 6.74 Mbps |
| R² | 0.976 |
| CV R² (mean ± std) | 0.935 ± 0.038 |
| Training time | 0.247 sec |
| Inference time | 0.017 ms/row |

---

### 3.6 PyTorch Feedforward MLP

**What it does:**
The PyTorch MLP is a fully connected neural network for tabular QoS prediction. It receives the same encoded BNN-UPC features as the XGBoost log-delay baseline and predicts `log_avg_delay`.

**Architecture used:**

```text
input_dim -> 128 -> 64 -> 32 -> 1
```

Each hidden layer uses ReLU activation and dropout. The model is trained with AdamW and mean squared error loss. Early stopping monitors a validation split taken from the training data, and the final holdout test set is evaluated only after training.

**Preprocessing:**
Numeric features are median-imputed and scaled with `StandardScaler`. Categorical features are most-frequent-imputed and one-hot encoded. The 11 raw BNN-UPC features become 15 encoded features after preprocessing.

**Why it is included:**
The MLP is the first deep-learning baseline in the project. It tests whether a simple neural network can match or improve on tree-based models for the tabular BNN-UPC QoS prediction task before moving to more complex sequence or graph neural models.

**Results on BNN-UPC `log_avg_delay`:**

| Metric | Value |
| --- | ---: |
| MAE | 0.185 |
| RMSE | 0.357 |
| R² | 0.766 |
| CV R² (mean ± std) | 0.769 ± 0.008 |
| Inference time | 0.00019 ms/row |

The MLP is currently competitive with XGBoost on the BNN-UPC log-delay target. Its CV R² is slightly higher than the XGBoost run, though the difference is small enough that both should be treated as comparable baselines rather than a decisive win.

---

## 4. Evaluation Metrics

### 4.1 MAE — Mean Absolute Error

**Formula:**
```
MAE = (1/n) × Σ |actual - predicted|
```

**What it means:**
The average absolute difference between the predicted and actual values, in the same unit as the target (Mbps). An MAE of 10.55 Mbps means the model's predictions are on average 10–11 Mbps away from the true measured throughput.

**Why it is useful:**
- Easy to interpret: the error is in the same unit as the target
- Treats all errors equally — a 5 Mbps error counts exactly half as much as a 10 Mbps error
- Less sensitive to large outliers than RMSE

---

### 4.2 RMSE — Root Mean Squared Error

**Formula:**
```
RMSE = √( (1/n) × Σ (actual - predicted)² )
```

**What it means:**
The square root of the average squared error, also in the target unit (Mbps). Because errors are squared before averaging, RMSE penalises large individual errors much more heavily than small ones.

**Why it is useful:**
- Highlights models that make occasional large mistakes, even if their average error is low
- Standard metric in network performance prediction because large prediction errors in SD-WAN can cause significant SLA violations
- Higher RMSE than MAE for the same model indicates that some predictions have large errors

---

### 4.3 R² — R-Squared (Coefficient of Determination)

**Formula:**
```
R² = 1 - (SS_residual / SS_total)

where:
  SS_residual = Σ (actual - predicted)²     ← model's error
  SS_total    = Σ (actual - mean(actual))²  ← baseline error (DummyRegressor)
```

**What it means:**
The proportion of variance in the target that the model explains. R² = 1.0 means perfect predictions. R² = 0.0 means the model does no better than predicting the training mean. R² < 0 means the model is worse than the DummyRegressor.

| R² value | Interpretation |
| --- | --- |
| 1.00 | Perfect predictions |
| 0.90–0.99 | Excellent — model explains most variance |
| 0.70–0.89 | Good |
| 0.50–0.69 | Moderate |
| < 0.50 | Poor |
| ≤ 0.00 | Worse than predicting the mean |

**Results in context:**
Linear Regression (R²=0.895) already explains 89.5% of variance in throughput. Random Forest (R²=0.980) explains 98%, confirming that non-linear interactions between radio configuration and offered load contribute meaningfully to the remaining variance.

---

### 4.4 CV R² — Cross-Validated R² Score

**What it is:**
Instead of evaluating on a single fixed 80/20 split, k-fold cross-validation (k=5 in this project) divides the training data into 5 equal folds. The model is trained on 4 folds and tested on the held-out fold. This is repeated 5 times, each time using a different fold as the test set. The CV R² is the mean R² across all 5 test folds.

**Why it matters more than holdout R²:**
The Zenodo dataset has only 604 rows. With such a small dataset, a single 80/20 split can give an optimistic or pessimistic result depending on which rows happen to land in the test set. CV R² averages over 5 different test sets and is therefore a more stable, trustworthy estimate of how the model will perform on unseen data.

In this project, there is a consistent gap of approximately 0.04–0.05 between holdout R² and CV R² for all models. This means the particular 80/20 random split produced a slightly favourable test set, and the CV R² is the more reliable number to report.

**CV standard deviation (CV R² std):**
The standard deviation across the 5 folds. A small std (e.g., 0.019 for Linear Regression) means consistent performance regardless of which rows are in the test set. A larger std (e.g., 0.044 for Random Forest) indicates that performance varies more between folds, which is expected when the dataset is small.

---

## 5. Full Results Comparison

### Zenodo Throughput Results

| Model | MAE (Mbps) | RMSE (Mbps) | R² | CV R² | CV R² std | Inference (ms/row) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DummyRegressor (mean) | 36.90 | 43.88 | -0.007 | -0.011 | 0.006 | 0.011 |
| Linear Regression | 10.55 | 14.15 | 0.895 | 0.851 | 0.019 | 0.011 |
| SVR (RBF kernel) | 5.69 | 8.76 | 0.960 | 0.924 | 0.028 | 0.022 |
| Random Forest | **2.82** | **6.21** | **0.980** | **0.936** | 0.044 | 0.121 |
| XGBoost | 3.29 | 6.74 | 0.976 | 0.935 | 0.038 | **0.017** |

**Key observations:**
- Random Forest achieves the best MAE and R² on both holdout and CV splits
- XGBoost achieves near-equivalent CV R² to Random Forest but is 7× faster at inference (0.017 ms vs 0.121 ms/row), making it more suitable for near-real-time SD-WAN deployment
- SVR performs significantly better than Linear Regression, confirming non-linear relationships in the data
- All four real models comfortably beat the DummyRegressor, confirming that the input features contain meaningful signal for predicting throughput
- CV R² is the primary metric for thesis reporting given the small dataset size (604 rows)

### BNN-UPC Log-Delay Results

| Model | MAE | RMSE | R² | CV R² | CV R² std | Inference (ms/row) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DummyRegressor (mean) | 0.538 | 0.740 | -0.001 | -0.000 | 0.000 | 0.00043 |
| Linear Regression | 0.227 | 0.415 | 0.685 | 0.690 | 0.014 | 0.00046 |
| SVR (RBF kernel) | **0.174** | 0.391 | 0.721 | 0.721 | 0.013 | 0.246 |
| Random Forest | 0.181 | 0.364 | 0.758 | 0.764 | 0.010 | 0.00703 |
| XGBoost | 0.185 | 0.359 | 0.765 | 0.766 | 0.008 | 0.00058 |
| PyTorch MLP | 0.185 | **0.357** | **0.766** | **0.769** | 0.008 | **0.00019** |

**Key observations:**
- The log transform materially improves model fit compared with raw `avg_delay`, because BNN-UPC delay is heavy-tailed under congestion.
- Routing and topology features are strong predictors. The BNN-UPC feature-importance files show `routing_hops`, queue weights, topology size, and ToS-related features among the strongest signals.
- The PyTorch MLP, XGBoost, and Random Forest are close on CV R². This suggests the current tabular feature representation is already informative, and future gains may require richer topology or temporal modelling rather than only larger tabular models.

## 6. Report File Behaviour

Training scripts append new timestamped rows to existing report CSVs instead of overwriting them. This applies to:

```text
src/train_baseline.py
src/train_zenodo_baseline.py
src/train_bnnupc_mlp.py
```

This means repeated runs preserve experiment history in the same report file. The `run_timestamp`, `dataset`, `model`, and `target` columns identify each run.
