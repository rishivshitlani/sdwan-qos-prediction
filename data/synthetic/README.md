# Synthetic SD-WAN QoS Dataset - Feature Definitions

This folder contains the generated synthetic dataset used to test the modelling pipeline before relying on public datasets.

Dataset file:

```text
sdwan_qos_synthetic.csv
```

## Columns

| Column | Meaning | Typical role |
|---|---|---|
| `latency_ms` | Simulated network latency in milliseconds. | Input feature. |
| `jitter_ms` | Simulated variation in packet delay in milliseconds. | Input feature in synthetic experiments. |
| `packet_loss_percent` | Simulated percentage of packets lost. | Input feature in synthetic experiments. |
| `bandwidth_utilization_percent` | Simulated percentage of available bandwidth already used. | Input feature. |
| `actual_throughput_mbps` | Simulated achieved throughput in Mbps. | Input feature in the original synthetic target rule. |
| `flow_duration_sec` | Simulated duration of a network flow in seconds. | Input feature. |
| `packet_count` | Simulated number of packets in the flow. | Input feature. |
| `protocol` | Simulated transport protocol, for example `TCP` or `UDP`. | Categorical input feature. |
| `application_type` | Simulated application category, such as `voice`, `video`, `web`, `file_transfer`, or `database`. | Categorical input feature. |
| `link_type` | Simulated WAN link type, such as `MPLS`, `Broadband`, or `LTE`. | Categorical input feature. |
| `time_of_day` | Simulated hour of day from 0 to 23. | Input feature. |
| `recommended_bandwidth_percent` | Rule-based bandwidth recommendation created by `generate_dataset.py`. | Supervised target for synthetic experiments. |

## Important Note

`recommended_bandwidth_percent` is not measured from a real SD-WAN system. It is created by a hand-written rule in `src/generate_dataset.py`. This is useful for pipeline testing, but it should be described as synthetic ground truth in reports and thesis writing.
