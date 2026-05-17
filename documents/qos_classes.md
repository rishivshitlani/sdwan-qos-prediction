# QoS Classes and Traffic Mappings

This document defines the initial QoS class policy used by the project. The mappings are intentionally explicit so that feature engineering, labelling, and thesis methodology can refer to one consistent classification rule.

The classes are ordered by expected service priority:

1. Gold: latency-sensitive and business-critical traffic
2. Silver: important interactive or transactional traffic
3. Bronze: best-effort, bulk, background, or unknown traffic

These definitions are a starting point for the capstone implementation. In a production SD-WAN deployment, the final policy would normally be validated against the organisation's application catalogue, SLA requirements, and security rules.

## Class Definitions

| QoS class | Priority | Typical SLA expectation | Description |
| --- | --- | --- | --- |
| Gold | Highest | Low latency, low jitter, low loss | Real-time and mission-critical traffic where delay or packet loss quickly affects user experience or service quality. |
| Silver | Medium | Stable latency and throughput | Business applications and interactive services that matter, but can tolerate more delay than real-time traffic. |
| Bronze | Lowest | Best effort | Bulk transfer, background, web browsing, unknown traffic, or traffic without strict QoS requirements. |

## Explicit Port and Protocol Mapping

Traffic should be mapped using the most specific available signal. If application identification is available, use it first. If not, fall back to protocol and port.

| QoS class | Application / traffic type | Protocol | Ports | Reason |
| --- | --- | --- | --- | --- |
| Gold | VoIP signalling | TCP/UDP | 5060, 5061 | SIP signalling for voice/video call setup. |
| Gold | Real-time media / RTP | UDP | 16384-32767 | Common RTP/RTCP media range used by voice and video platforms. |
| Gold | DNS | UDP/TCP | 53 | Small but latency-sensitive service dependency. Slow DNS affects most applications. |
| Gold | NTP | UDP | 123 | Time synchronisation dependency for distributed systems. |
| Gold | SD-WAN control / tunnel keepalive | UDP | 500, 4500 | IPSec/IKE and NAT traversal are often part of encrypted WAN tunnels. |
| Silver | HTTPS web applications | TCP | 443 | Interactive business web applications and APIs. |
| Silver | HTTP web applications | TCP | 80 | Interactive web traffic where HTTPS classification is unavailable. |
| Silver | SSH administration | TCP | 22 | Interactive administrative access. |
| Silver | Remote desktop | TCP/UDP | 3389 | Interactive remote access. |
| Silver | Database services | TCP | 1433, 1521, 3306, 5432 | Common SQL Server, Oracle, MySQL, and PostgreSQL application traffic. |
| Silver | Email submission / retrieval | TCP | 25, 465, 587, 993, 995 | Business communication traffic that is important but not real-time. |
| Bronze | Bulk file transfer | TCP | 20, 21, 989, 990 | FTP/FTPS transfers can consume bandwidth and normally tolerate delay. |
| Bronze | SMB file sharing | TCP | 445 | File sharing and bulk copy traffic. |
| Bronze | Backup / rsync-style transfer | TCP | 873 | Background synchronisation and backup traffic. |
| Bronze | Peer-to-peer or unknown high ports | TCP/UDP | 49152-65535 | Treat as best effort unless application identification proves otherwise. |
| Bronze | Unknown or unmapped traffic | TCP/UDP/Other | Any | Conservative default when no reliable business or real-time classification exists. |

## Fallback Rules

Use the following rules when traffic does not exactly match the table above:

| Rule order | Condition | QoS class |
| --- | --- | --- |
| 1 | Known real-time voice/video, tunnel control, DNS, or NTP | Gold |
| 2 | Known business application, administration, database, email, or interactive web traffic | Silver |
| 3 | Bulk transfer, backup, file sharing, unknown, or unmapped traffic | Bronze |

## Suggested Numeric Encoding

Some models require numeric labels. Use this encoding only after preserving the original class name in a separate column:

| QoS class | Numeric encoding |
| --- | ---: |
| Gold | 3 |
| Silver | 2 |
| Bronze | 1 |

The numeric values represent priority order. They should not be interpreted as measured bandwidth percentages or as ground-truth QoS outcomes.

## BNN-UPC ToS Mapping

The BNN-UPC simulation workflow uses BNNetSimulator's `ToS` field to encode the same three QoS classes:

| BNNetSimulator ToS | QoS class | Priority | Scenario A queue weight |
| ---: | --- | --- | ---: |
| 0 | Gold | Highest | 60% |
| 1 | Silver | Medium | 30% |
| 2 | Bronze | Lowest | 10% |

This mapping is used by:

```text
src/generate_bnnupc_dataset.py
src/process_bnnupc_dataset.py
src/train_bnnupc_mlp.py
```

In the processed BNN-UPC dataset, both `tos` and `qos_class` are retained. Keeping both columns makes the numeric simulator encoding traceable while preserving the human-readable QoS label for analysis and one-hot model features.

## Relationship to Option 2 Targets

For the Option 2 research direction, QoS classes can be used as constraints or weights in an optimisation-based target. For example, a target can be derived by assigning higher penalty weights to Gold traffic SLA violations than to Silver or Bronze violations.

An initial weighting scheme can be:

| QoS class | SLA violation penalty weight |
| --- | ---: |
| Gold | 3.0 |
| Silver | 2.0 |
| Bronze | 1.0 |

This supports a derived target such as `recommended_bandwidth_percent` or a future `sla_violation_risk_score`, while keeping the distinction clear between observed QoS measurements and optimisation-derived labels.
