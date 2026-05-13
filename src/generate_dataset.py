from pathlib import Path

import numpy as np
import pandas as pd

# Keep the synthetic dataset reproducible across runs.
np.random.seed(42)

# Number of synthetic network flow records to generate.
n = 50000

# Categorical values used to mimic SD-WAN traffic and link context.
application_types = ["voice", "video", "web", "file_transfer", "database"]
protocols = ["TCP", "UDP"]
link_types = ["MPLS", "Broadband", "LTE"]

# Generate a synthetic SD-WAN-like flow table. The numeric ranges are broad
# approximations of network conditions rather than measurements from real links.
data = pd.DataFrame({
    "latency_ms": np.random.uniform(5, 250, n),
    "jitter_ms": np.random.uniform(0, 80, n),
    "packet_loss_percent": np.random.uniform(0, 8, n),
    "bandwidth_utilization_percent": np.random.uniform(10, 100, n),
    "actual_throughput_mbps": np.random.uniform(1, 500, n),
    "flow_duration_sec": np.random.uniform(1, 3600, n),
    "packet_count": np.random.randint(10, 100000, n),
    "protocol": np.random.choice(protocols, n),
    "application_type": np.random.choice(application_types, n),
    "link_type": np.random.choice(link_types, n),
    "time_of_day": np.random.randint(0, 24, n)
})


def calculate_bandwidth(row):
    """Create a rule-based QoS bandwidth recommendation for one flow."""
    # Start with a neutral baseline, then add pressure based on application
    # importance and observed network quality.
    base = 30

    # Give latency-sensitive or business-critical applications more bandwidth.
    if row["application_type"] == "voice":
        base += 30
    elif row["application_type"] == "video":
        base += 25
    elif row["application_type"] == "database":
        base += 15
    elif row["application_type"] == "web":
        base += 10
    else:
        base += 5

    # Increase the recommendation when network conditions look degraded.
    if row["latency_ms"] > 100:
        base += 10

    if row["jitter_ms"] > 30:
        base += 10

    if row["packet_loss_percent"] > 2:
        base += 10

    if row["bandwidth_utilization_percent"] > 80:
        base += 5

    # Cap the recommendation because the target is a percentage.
    return min(base, 100)


# This is the supervised regression target for the synthetic dataset.
data["recommended_bandwidth_percent"] = data.apply(calculate_bandwidth, axis=1)

output_path = Path("data/synthetic/sdwan_qos_synthetic.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)

# Save the generated data. The CSV is ignored by Git because it is generated.
data.to_csv(output_path, index=False)

print(f"Dataset created: {output_path}")
print(data.head())
