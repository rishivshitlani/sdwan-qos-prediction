"""Create an SD-WAN-style ML sample from CICIDS2017 flow CSV files.

The output aligns CICIDS2017 flow features with the project feature set:
latency, jitter, packet loss, bandwidth utilization, throughput, flow duration,
packet count, protocol, application type, link type, time of day, and the
recommended bandwidth target.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "cicids" / "MachineLearningCVE"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "cicids2017_project_aligned_sample.csv"

# Raw CICIDS columns needed to build the project-style feature table.
RAW_COLUMNS = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Flow Bytes/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "RST Flag Count",
    "Label",
]

# The first columns in the output match the features named in the project README.
PROJECT_FEATURE_COLUMNS = [
    "latency_ms",
    "jitter_ms",
    "packet_loss_percent",
    "bandwidth_utilization_percent",
    "throughput_mbps",
    "flow_duration_sec",
    "packet_count",
    "protocol",
    "application_type",
    "link_type",
    "time_of_day",
]

TARGET_COLUMN = "recommended_bandwidth_percent"

# Extra fields retained so rows can still be traced back to CICIDS labels/files.
TRACEABILITY_COLUMNS = [
    "cicids_attack_label",
    "is_attack",
    "source_file",
    "destination_port",
    "byte_count",
]


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove leading/trailing spaces from CICIDS column names."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    return df


def clean_labels(labels: pd.Series) -> pd.Series:
    """Normalize CICIDS label text, including the odd replacement character."""
    return labels.astype(str).str.strip().str.replace("\ufffd", "-", regex=False)


def infer_protocol(destination_port: pd.Series) -> pd.Series:
    """Infer TCP/UDP from common destination ports when protocol is unavailable."""
    ports = pd.to_numeric(destination_port, errors="coerce").fillna(-1).astype(int)
    udp_like_ports = [53, 67, 68, 69, 123, 161, 162, 500, 4500, 5060, 5061]
    return pd.Series(np.where(ports.isin(udp_like_ports), "UDP", "TCP"), index=destination_port.index)


def infer_application_type(destination_port: pd.Series) -> pd.Series:
    """Map well-known destination ports into the project's traffic categories."""
    ports = pd.to_numeric(destination_port, errors="coerce").fillna(-1).astype(int)

    conditions = [
        ports.isin([80, 443, 8080, 8443]),
        ports.isin([554, 1935, 5004, 5005, 8081]),
        ports.isin([20, 21]),
        ports.isin([25, 110, 143, 465, 587, 993, 995]),
        ports.isin([53]),
        ports.isin([22]),
        ports.isin([5060, 5061]),
        ports.isin([3306, 5432, 1433, 1521, 27017]),
    ]
    choices = ["web", "video", "file_transfer", "mail", "dns", "ssh", "voice", "database"]

    return pd.Series(np.select(conditions, choices, default="other"), index=destination_port.index)


def calculate_bandwidth(row: pd.Series) -> int:
    """Create a heuristic QoS bandwidth target for one CICIDS-derived flow."""
    base = 30

    # Start from application priority, similar to the synthetic dataset logic.
    if row["application_type"] == "voice":
        base += 30
    elif row["application_type"] == "video":
        base += 25
    elif row["application_type"] == "database":
        base += 15
    elif row["application_type"] == "web":
        base += 10
    elif row["application_type"] == "file_transfer":
        base += 5
    else:
        base += 8

    # Add bandwidth pressure for degraded flow conditions.
    if row["latency_ms"] > 100:
        base += 10
    if row["jitter_ms"] > 30:
        base += 10
    if row["packet_loss_percent"] > 2:
        base += 10
    if row["bandwidth_utilization_percent"] > 80:
        base += 5

    # Attack traffic should not receive the same QoS priority as benign traffic.
    if row["is_attack"]:
        base -= 10

    return max(0, min(base, 100))


def create_sdwan_style_features(
    df: pd.DataFrame,
    source_file: str,
    assumed_link_capacity_mbps: float,
) -> pd.DataFrame:
    """Convert one raw CICIDS chunk into project-aligned SD-WAN features."""
    df = clean_columns(df)
    df = df.replace([np.inf, -np.inf], np.nan)

    # Combine forward/backward packet and byte counts into total flow values.
    total_bytes = df["Total Length of Fwd Packets"] + df["Total Length of Bwd Packets"]
    packet_count = df["Total Fwd Packets"] + df["Total Backward Packets"]

    # CICIDS Flow Bytes/s is bytes per second; multiply by 8 for bits and
    # divide by 1,000,000 to express it as Mbps.
    throughput_mbps = df["Flow Bytes/s"] * 8 / 1_000_000

    # CICIDS2017 does not directly contain SD-WAN QoS fields such as latency,
    # jitter, packet loss, link type, or time of day. The fields below are
    # proxies/placeholders so this sample can be used with the project pipeline.
    processed = pd.DataFrame(
        {
            "latency_ms": df["Flow IAT Mean"] / 1_000,
            "jitter_ms": df["Flow IAT Std"] / 1_000,
            "packet_loss_percent": np.where(df["RST Flag Count"] > 0, 1.0, 0.0),
            "bandwidth_utilization_percent": (throughput_mbps / assumed_link_capacity_mbps * 100).clip(0, 100),
            "throughput_mbps": throughput_mbps,
            "flow_duration_sec": df["Flow Duration"] / 1_000_000,
            "packet_count": packet_count,
            "protocol": infer_protocol(df["Destination Port"]),
            "application_type": infer_application_type(df["Destination Port"]),
            "link_type": "unknown",
            "time_of_day": -1,
            "byte_count": total_bytes,
            "destination_port": df["Destination Port"],
            "cicids_attack_label": clean_labels(df["Label"]),
            "source_file": source_file,
        }
    )

    # Keep the original CICIDS label as context, plus a simple binary flag.
    processed["is_attack"] = (processed["cicids_attack_label"].str.upper() != "BENIGN").astype(int)

    # This is a derived target for experiments, not a real CICIDS QoS label.
    processed["recommended_bandwidth_percent"] = processed.apply(calculate_bandwidth, axis=1)

    output_columns = PROJECT_FEATURE_COLUMNS + [TARGET_COLUMN] + TRACEABILITY_COLUMNS
    return processed[output_columns].replace([np.inf, -np.inf], np.nan).dropna()


def get_csv_files(input_dir: Path) -> list[Path]:
    """Return all raw CICIDS CSV files in a stable order."""
    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")
    return csv_files


def build_project_aligned_sample(
    input_dir: Path,
    output_path: Path,
    sample_size: int,
    chunksize: int,
    assumed_link_capacity_mbps: float,
) -> None:
    """Build and save a balanced-by-file project-aligned sample."""
    csv_files = get_csv_files(input_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Sample evenly from each day/file so the output is not dominated by the
    # first CSV encountered.
    per_file_sample_size = max(1, sample_size // len(csv_files))
    sampled_chunks = []
    rows_seen = 0

    print(f"Reading CICIDS2017 CSV files from: {input_dir}")

    for csv_file in csv_files:
        print(f"Extracting SD-WAN-style features from {csv_file.name}")
        sampled_for_file = 0

        # Only load the raw columns needed for project feature generation.
        for chunk in pd.read_csv(
            csv_file,
            chunksize=chunksize,
            usecols=lambda col: col.strip() in RAW_COLUMNS,
        ):
            rows_seen += len(chunk)
            remaining = per_file_sample_size - sampled_for_file
            if remaining <= 0:
                continue

            features = create_sdwan_style_features(
                chunk,
                source_file=csv_file.name,
                assumed_link_capacity_mbps=assumed_link_capacity_mbps,
            )
            take = min(remaining, len(features))
            if take <= 0:
                continue

            # Deterministic sampling makes repeated runs comparable.
            sampled_chunks.append(features.sample(n=take, random_state=42))
            sampled_for_file += take

    project_sample = pd.concat(sampled_chunks, ignore_index=True)
    project_sample.to_csv(output_path, index=False)

    print(f"\nRows scanned: {rows_seen:,}")
    print(f"Sample rows written: {len(project_sample):,}")
    print(f"Output written to: {output_path}")
    print("\nOutput columns:")
    for column in project_sample.columns:
        print(f"- {column}")


def parse_args() -> argparse.Namespace:
    """Read command-line options for the sample creation script."""
    parser = argparse.ArgumentParser(
        description="Create an SD-WAN-style project sample from CICIDS2017 flow CSV files."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--sample-size", type=int, default=100_000)
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--assumed-link-capacity-mbps", type=float, default=1_000.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_project_aligned_sample(
        input_dir=args.input_dir,
        output_path=args.output_path,
        sample_size=args.sample_size,
        chunksize=args.chunksize,
        assumed_link_capacity_mbps=args.assumed_link_capacity_mbps,
    )
