# Zenodo 5G Campus Network Dataset - Feature Definitions

This folder contains the Zenodo record `13754300` dataset used for the Zenodo-first QoS experiments. It includes throughput measurements and packet-level one-way-delay measurements from two 5G testbeds: NTNU and WUE.

## Quick Terms

| Term | Meaning |
|---|---|
| OWD | One-Way Delay. The time taken for a packet to travel from sender to receiver in one direction. |
| IAT | Inter-Arrival Time. The time gap between one packet arriving and the next packet arriving. |
| gNB | 5G base station / radio access network node. In this dataset the implementation is `oai` or `srs`. |
| SDR | Software Defined Radio hardware used in the experiment, for example `b200`. |
| Uplink | Traffic direction from user equipment/client side toward the network. |
| Downlink | Traffic direction from network toward the user equipment/client side. |
| GTP | GPRS Tunnelling Protocol, used in mobile networks to encapsulate user traffic. |
| Offered throughput | Traffic rate requested/generated during the experiment. |
| Actual throughput | Traffic rate successfully delivered/measured during the experiment. |

## Files

| File pattern | Meaning |
|---|---|
| `*_tput_all_Throughput.csv` | iperf3 throughput summary files. These are the main source for the first baseline model. |
| `*_owd_Packets_with_IATs.csv` | Packet-level OWD files with inter-arrival time already calculated. These are used for optional/advanced latency aggregation. |
| `*_owd_all_Packets.csv` | Packet-level OWD files. These are more raw packet records and mostly duplicate the packet-level measurement purpose of the IAT files. |

## Throughput File Columns

| Column | Meaning | Typical use |
|---|---|---|
| `mbpsactual_uplink` | Measured uplink throughput in Mbps. | QoS output/target candidate. |
| `mbpsactual_downlink` | Measured downlink throughput in Mbps. | QoS output/target candidate. |
| `meanjitterms_uplink` | Mean uplink jitter in milliseconds. | QoS outcome. Usually dropped from first pre-run baseline inputs. |
| `meanjitterms_downlink` | Mean downlink jitter in milliseconds. | QoS outcome. Usually dropped from first pre-run baseline inputs. |
| `meanloss_uplink` | Mean uplink packet loss ratio. | QoS outcome. |
| `meanloss_downlink` | Mean downlink packet loss ratio. | QoS outcome. |
| `gnb` | 5G gNB implementation used, such as `oai` or `srs`. | Input feature. |
| `sdr` | SDR hardware used, such as `b200`. | Input feature. |
| `bw` | Configured radio bandwidth in MHz. | Input feature, renamed to `bw_mhz` in processed data. |
| `slots` | Configured number of downlink slots. | Input feature. |
| `ratio` | Configured downlink/uplink slot ratio. | Input feature. |
| `mbpsoffered_uplink` | Offered/generated uplink throughput in Mbps. | Input feature after unpivoting. |
| `mbpsoffered_downlink` | Offered/generated downlink throughput in Mbps. | Input feature after unpivoting. |
| `rep` | Repetition number of the measurement. | Trace/config feature. |
| `scenario` | Encoded measurement scenario string. | Traceability; not currently used directly for modelling. |

## OWD Packet File Columns

| Column | Meaning | Typical use |
|---|---|---|
| `src` | Source device identifier. | Traceability. |
| `Timestamp` | Packet timestamp from Wireshark/capture tooling. | Used to derive time features. |
| `SourceIPOuter` | Source IP address of the outer GTP packet. | Packet trace metadata. |
| `DestinationIPOuter` | Destination IP address of the outer GTP packet. | Packet trace metadata. |
| `SourceIPInner` | Source IP address of the inner user packet. | Packet trace metadata. |
| `DestinationIPInner` | Destination IP address of the inner user packet. | Packet trace metadata. |
| `PacketSize` | Packet size. | Packet-level traffic size feature. |
| `SeqNum` | Sequence number injected during traffic generation. | Packet ordering/traceability. |
| `iat` | Inter-arrival time between packets. `iat == 0` usually marks the first packet of a burst. | Aggregated to latency-style features. |
| `trel` | Relative timestamp within a measurement run. | Packet timing feature. |
| `gnb` | 5G gNB implementation. | Join/config key. |
| `sdr` | SDR hardware. | Join/config key. |
| `bw` | Configured radio bandwidth in MHz. | Join/config key. |
| `slots` | Configured number of downlink slots. | Join/config key. |
| `ratio` | Configured downlink/uplink slot ratio. | Join/config key. |
| `pdist` | Packet generation distribution, such as deterministic or negative-exponential. | Traffic generation setting. |
| `piat` | Configured packet generation inter-arrival time. | Traffic generation setting. |
| `psize` | Configured packet size mode. | Traffic generation setting. |
| `pnpak` | Total number of packets generated for the measurement run. | Traffic generation setting. |
| `rep` | Repetition number of the measurement. | Trace/config feature. |
| `scenario` | Encoded measurement scenario string. | Traceability. |
| `direction` | Traffic direction, usually uplink or downlink. | Join/config key. |

## Processed Project-Aligned Columns

These columns are created by `src/process_zenodo_dataset.py` in `data/processed/zenodo_13754300_project_aligned.csv`.

| Column | Meaning |
|---|---|
| `latency_ms` | Aggregated latency-style value from OWD/IAT files. In quick `--skip-owd` mode this is `-1`. |
| `jitter_ms` | Jitter from the throughput summary file. |
| `packet_loss_percent` | Packet loss converted from raw mean loss ratio to percent. |
| `bandwidth_utilization_percent` | Offered throughput divided by assumed link capacity. |
| `actual_throughput_mbps` | Actual measured throughput. Recommended first Zenodo target. |
| `flow_duration_sec` | Aggregated duration estimate from OWD/IAT files. In quick mode this is `-1`. |
| `packet_count` | Aggregated packet count from OWD/IAT files. In quick mode this is `-1`. |
| `protocol` | Set to `UDP` because the throughput tests use UDP-style generated traffic. |
| `application_type` | Set to `throughput_test`. |
| `link_type` | Set to `5G`. |
| `time_of_day` | Hour derived from packet timestamp when OWD is included. In quick mode this is `-1`. |
| `recommended_bandwidth_percent` | Derived target: `actual_throughput_mbps / offered_throughput_mbps * 100`, clipped to 0-100. |
| `testbed` | Testbed source, `ntnu` or `wue`. |
| `gnb` | 5G gNB implementation. |
| `sdr` | SDR hardware. |
| `bw_mhz` | Radio bandwidth in MHz. |
| `slots` | Configured downlink slots. |
| `ratio` | Configured downlink/uplink ratio. |
| `rep` | Measurement repetition. |
| `direction` | Uplink or downlink. |
| `offered_throughput_mbps` | Offered/generated throughput. Key controllable input feature. |
| `source_file` | Raw throughput file the row came from. Traceability only. |

## Modelling Note

For the first Zenodo baseline, `actual_throughput_mbps` is the preferred target. If `recommended_bandwidth_percent` is used as the target, `actual_throughput_mbps` must be dropped from model inputs because the recommended bandwidth value is derived from actual throughput and offered throughput.
