# AI-Driven QoS Prediction in SD-WAN Networks

This repository contains the implementation work for my MSc AI capstone project: **AI-Driven QoS Prediction in SD-WAN Networks**.

The project explores how machine learning models can be used to predict recommended QoS bandwidth allocation in an SD-WAN-like environment using network flow and link-level features.

## Project Aim

The aim of this project is to evaluate whether machine learning models can predict QoS-related bandwidth allocation decisions based on network conditions such as latency, jitter, packet loss, throughput, application type, protocol, and link type.

The prediction task is treated as a **supervised regression problem**.

## Research Questions

This project follows the research questions from the thesis proposal:

1. What types of AI or ML techniques are currently or can be used for predicting QoS in SD-WAN?
2. What are the benefits and limitations of these approaches in terms of prediction accuracy, scalability, and real-world use?
3. How can QoS prediction be effectively integrated into SD-WAN production environments to improve network performance?

## Repository Structure

```text
sdwan-qos-prediction/
|
├── data/
│   └── sdwan_qos_synthetic.csv
|
├── src/
│   ├── generate_dataset.py
│   └── train_baseline.py
|
├── notebooks/
|
├── reports/
│   └── methodology_notes.md
|
├── figures/
|
├── models/
|
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset

The project currently uses a synthetic SD-WAN-like dataset.

Each row represents a network traffic condition or flow. The dataset includes the following features:

| Feature | Description |
| --- | --- |
| latency_ms | Network latency in milliseconds |
| jitter_ms | Variation in packet delay |
| packet_loss_percent | Percentage of packets lost |
| bandwidth_utilization_percent | Current link utilisation |
| throughput_mbps | Measured throughput in Mbps |
| flow_duration_sec | Duration of the traffic flow |
| packet_count | Number of packets in the flow |
| protocol | Transport protocol, such as TCP or UDP |
| application_type | Type of traffic, such as voice, video, web, file transfer, or database |
| link_type | Underlay link type, such as MPLS, Broadband, or LTE |
| time_of_day | Hour of day when the flow occurs |

The target variable is:

| Target | Description |
| --- | --- |
| recommended_bandwidth_percent | Predicted or recommended QoS bandwidth allocation percentage |

## Machine Learning Approach

The first implementation uses baseline supervised regression models:

* Linear Regression
* Random Forest Regressor

The models are trained to predict `recommended_bandwidth_percent`.

Categorical features are one-hot encoded, and numerical features are standardised before model training.

## Evaluation Metrics

The models are evaluated using:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score
* Inference Time

These metrics help compare prediction accuracy and practical suitability for near-real-time SD-WAN use.

## Setup Instructions

1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/sdwan-qos-prediction.git
cd sdwan-qos-prediction
```

2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

For Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

## How to Run

1. Generate the synthetic dataset

```bash
python src/generate_dataset.py
```

This creates:

```text
data/sdwan_qos_synthetic.csv
```

2. Train baseline models

```bash
python src/train_baseline.py
```

This trains and evaluates the baseline models.

Example output:

```text
Model: Linear Regression
MAE: ...
RMSE: ...
R2 Score: ...
Inference Time: ...

Model: Random Forest
MAE: ...
RMSE: ...
R2 Score: ...
Inference Time: ...
```

## Current Status

Current implementation includes:

* Synthetic SD-WAN-like dataset generation
* Baseline ML training pipeline
* Linear Regression model
* Random Forest model
* Evaluation using MAE, RMSE, R², and inference time

## Planned Improvements

Planned next steps:

* Add Gradient Boosting or XGBoost model
* Add Support Vector Regression
* Add feature importance analysis
* Add visualisations for model comparison
* Add error analysis
* Improve dataset generation logic
* Add thesis-ready experiment results and figures

## Notes on Data

This project currently uses synthetic data because real SD-WAN QoS datasets are difficult to obtain publicly. The synthetic data is designed to represent realistic SD-WAN-like network conditions, but it does not fully capture all production network behaviours.

This limitation will be clearly discussed in the thesis.

## Author
Rishiv Shitlani
MSc Computer Science Artificial Intelligence
University of Galway
