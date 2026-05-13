# CICIDS2017 Dataset - Feature Definitions

This folder contains CICIDS2017 flow CSV files under:

```text
data/raw/cicids/MachineLearningCVE/
```

CICIDS2017 is an intrusion-detection dataset. It is useful for exploratory network-flow processing, but it is not a real SD-WAN QoS dataset. QoS-style fields created from CICIDS are proxy features only.

## Quick Terms

| Term | Meaning |
|---|---|
| Flow | A group of packets sharing a communication direction/session. |
| Fwd | Forward direction of the flow. |
| Bwd | Backward direction of the flow. |
| IAT | Inter-Arrival Time between packets in a flow. |
| PSH | TCP Push flag. |
| URG | TCP Urgent flag. |
| FIN | TCP Finish flag. |
| SYN | TCP Synchronize flag used when starting a TCP connection. |
| RST | TCP Reset flag. |
| ACK | TCP Acknowledgement flag. |
| CWE | Congestion Window Reduced flag. |
| ECE | ECN-Echo flag. |
| Bulk | A group of packets/bytes transferred together in one direction. |
| Active | Time period when the flow is actively sending packets. |
| Idle | Time period when the flow is not sending packets. |

## Raw CICIDS2017 Columns

| Column | Meaning |
|---|---|
| `Destination Port` | Destination transport-layer port number. |
| `Flow Duration` | Total duration of the flow, usually in microseconds. |
| `Total Fwd Packets` | Total packets in the forward direction. |
| `Total Backward Packets` | Total packets in the backward direction. |
| `Total Length of Fwd Packets` | Total bytes/length of packets in the forward direction. |
| `Total Length of Bwd Packets` | Total bytes/length of packets in the backward direction. |
| `Fwd Packet Length Max` | Maximum forward packet length. |
| `Fwd Packet Length Min` | Minimum forward packet length. |
| `Fwd Packet Length Mean` | Mean forward packet length. |
| `Fwd Packet Length Std` | Standard deviation of forward packet length. |
| `Bwd Packet Length Max` | Maximum backward packet length. |
| `Bwd Packet Length Min` | Minimum backward packet length. |
| `Bwd Packet Length Mean` | Mean backward packet length. |
| `Bwd Packet Length Std` | Standard deviation of backward packet length. |
| `Flow Bytes/s` | Average bytes transferred per second for the whole flow. |
| `Flow Packets/s` | Average packets transferred per second for the whole flow. |
| `Flow IAT Mean` | Mean inter-arrival time across packets in the flow. |
| `Flow IAT Std` | Standard deviation of flow inter-arrival time. |
| `Flow IAT Max` | Maximum flow inter-arrival time. |
| `Flow IAT Min` | Minimum flow inter-arrival time. |
| `Fwd IAT Total` | Total inter-arrival time in the forward direction. |
| `Fwd IAT Mean` | Mean forward inter-arrival time. |
| `Fwd IAT Std` | Standard deviation of forward inter-arrival time. |
| `Fwd IAT Max` | Maximum forward inter-arrival time. |
| `Fwd IAT Min` | Minimum forward inter-arrival time. |
| `Bwd IAT Total` | Total inter-arrival time in the backward direction. |
| `Bwd IAT Mean` | Mean backward inter-arrival time. |
| `Bwd IAT Std` | Standard deviation of backward inter-arrival time. |
| `Bwd IAT Max` | Maximum backward inter-arrival time. |
| `Bwd IAT Min` | Minimum backward inter-arrival time. |
| `Fwd PSH Flags` | Count of TCP PSH flags in the forward direction. |
| `Bwd PSH Flags` | Count of TCP PSH flags in the backward direction. |
| `Fwd URG Flags` | Count of TCP URG flags in the forward direction. |
| `Bwd URG Flags` | Count of TCP URG flags in the backward direction. |
| `Fwd Header Length` | Total header length in the forward direction. |
| `Bwd Header Length` | Total header length in the backward direction. |
| `Fwd Packets/s` | Forward packets per second. |
| `Bwd Packets/s` | Backward packets per second. |
| `Min Packet Length` | Minimum packet length in the flow. |
| `Max Packet Length` | Maximum packet length in the flow. |
| `Packet Length Mean` | Mean packet length in the flow. |
| `Packet Length Std` | Standard deviation of packet length. |
| `Packet Length Variance` | Variance of packet length. |
| `FIN Flag Count` | Count of TCP FIN flags. |
| `SYN Flag Count` | Count of TCP SYN flags. |
| `RST Flag Count` | Count of TCP RST flags. |
| `PSH Flag Count` | Count of TCP PSH flags. |
| `ACK Flag Count` | Count of TCP ACK flags. |
| `URG Flag Count` | Count of TCP URG flags. |
| `CWE Flag Count` | Count of TCP CWE flags. |
| `ECE Flag Count` | Count of TCP ECE flags. |
| `Down/Up Ratio` | Ratio of download/backward traffic to upload/forward traffic. |
| `Average Packet Size` | Average packet size across the flow. |
| `Avg Fwd Segment Size` | Average TCP segment size in the forward direction. |
| `Avg Bwd Segment Size` | Average TCP segment size in the backward direction. |
| `Fwd Header Length.1` | Duplicate/alternate forward header length column present in CICIDS files. |
| `Fwd Avg Bytes/Bulk` | Average bytes per forward bulk transfer. |
| `Fwd Avg Packets/Bulk` | Average packets per forward bulk transfer. |
| `Fwd Avg Bulk Rate` | Average forward bulk transfer rate. |
| `Bwd Avg Bytes/Bulk` | Average bytes per backward bulk transfer. |
| `Bwd Avg Packets/Bulk` | Average packets per backward bulk transfer. |
| `Bwd Avg Bulk Rate` | Average backward bulk transfer rate. |
| `Subflow Fwd Packets` | Forward packets counted within subflows. |
| `Subflow Fwd Bytes` | Forward bytes counted within subflows. |
| `Subflow Bwd Packets` | Backward packets counted within subflows. |
| `Subflow Bwd Bytes` | Backward bytes counted within subflows. |
| `Init_Win_bytes_forward` | Initial TCP window size in the forward direction. |
| `Init_Win_bytes_backward` | Initial TCP window size in the backward direction. |
| `act_data_pkt_fwd` | Number of forward packets carrying actual data payload. |
| `min_seg_size_forward` | Minimum observed forward TCP segment size. |
| `Active Mean` | Mean active duration of the flow. |
| `Active Std` | Standard deviation of active duration. |
| `Active Max` | Maximum active duration. |
| `Active Min` | Minimum active duration. |
| `Idle Mean` | Mean idle duration of the flow. |
| `Idle Std` | Standard deviation of idle duration. |
| `Idle Max` | Maximum idle duration. |
| `Idle Min` | Minimum idle duration. |
| `Label` | CICIDS traffic label, such as `BENIGN` or an attack category. |

## Project-Aligned CICIDS Proxy Columns

`src/create_cicids_project_aligned_sample.py` creates a project-style sample from CICIDS. These fields are proxies, not true SD-WAN QoS measurements.

| Column | Meaning |
|---|---|
| `latency_ms` | Proxy derived from `Flow IAT Mean / 1000`. This is not true RTT latency. |
| `jitter_ms` | Proxy derived from `Flow IAT Std / 1000`. |
| `packet_loss_percent` | Proxy set to `1.0` when `RST Flag Count > 0`, otherwise `0.0`. This is not measured packet loss. |
| `bandwidth_utilization_percent` | `actual_throughput_mbps / assumed_link_capacity_mbps * 100`, clipped to 0-100. |
| `actual_throughput_mbps` | Derived from `Flow Bytes/s * 8 / 1,000,000`. |
| `flow_duration_sec` | Derived from `Flow Duration / 1,000,000`. |
| `packet_count` | `Total Fwd Packets + Total Backward Packets`. |
| `protocol` | Port-based approximation of `TCP` or `UDP`. |
| `application_type` | Port-based category such as web, video, file transfer, mail, DNS, SSH, voice, database, or other. |
| `link_type` | Set to `unknown` because CICIDS has no SD-WAN link type. |
| `time_of_day` | Set to `-1` because the selected CICIDS columns do not provide time-of-day. |
| `recommended_bandwidth_percent` | Heuristic target generated by project code, not a CICIDS label. |
| `cicids_attack_label` | Original CICIDS label. |
| `is_attack` | Binary flag derived from `cicids_attack_label != BENIGN`. |
| `source_file` | Raw CICIDS CSV file source. |
| `destination_port` | Original destination port retained for traceability. |
| `byte_count` | Combined forward and backward byte count. |

## Modelling Note

CICIDS2017 should not be presented as a true SD-WAN QoS dataset. It can support exploratory feature engineering and pipeline testing, but QoS fields derived from it should be described as proxy features.
