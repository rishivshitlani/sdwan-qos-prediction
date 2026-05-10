import pandas as pd
import numpy as np

np.random.seed(42)

n = 5000

application_types = ["voice", "video", "web", "file_transfer", "database"]
protocols = ["TCP", "UDP"]
link_types = ["MPLS", "Broadband", "LTE"]

data = pd.DataFrame({
    "latency_ms": np.random.uniform(5, 250, n),
    "jitter_ms": np.random.uniform(0, 80, n),
    "packet_loss_percent": np.random.uniform(0, 8, n),
    "bandwidth_utilization_percent": np.random.uniform(10, 100, n),
    "throughput_mbps": np.random.uniform(1, 500, n),
    "flow_duration_sec": np.random.uniform(1, 3600, n),
    "packet_count": np.random.randint(10, 100000, n),
    "protocol": np.random.choice(protocols, n),
    "application_type": np.random.choice(application_types, n),
    "link_type": np.random.choice(link_types, n),
    "time_of_day": np.random.randint(0, 24, n)
})


def calculate_bandwidth(row):
    base = 30

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

    if row["latency_ms"] > 100:
        base += 10

    if row["jitter_ms"] > 30:
        base += 10

    if row["packet_loss_percent"] > 2:
        base += 10

    if row["bandwidth_utilization_percent"] > 80:
        base += 5

    return min(base, 100)

data["recommended_bandwidth_percent"] = data.apply(calculate_bandwidth, axis=1)

data.to_csv("data/sdwan_qos_synthetic.csv", index=False)

print("Dataset created: data/sdwan_qos_synthetic.csv")
print(data.head())
