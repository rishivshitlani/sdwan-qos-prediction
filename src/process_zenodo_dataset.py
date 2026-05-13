"""Create an SD-WAN-style ML sample from the Zenodo 5G campus network dataset.

The dataset (Zenodo record 13754300) contains iperf3 throughput measurements
and one-way delay (OWD) packet captures from two 5G testbeds: NTNU (Norway)
and WUE (Würzburg, Germany).

Throughput files provide jitter, packet loss, and actual/offered throughput
per measurement run. OWD packet files provide per-packet inter-arrival times
which are aggregated per radio configuration to derive latency statistics.

The scenario strings in the two file types differ (OWD encodes packet
generation params; throughput encodes offered load), so they cannot be joined
on scenario. The join key is the shared 5G radio configuration:
    gnb, sdr, bw, slots, ratio, direction

OWD aggregation averages across all reps per radio config to produce one
latency estimate per configuration, which is then matched to every throughput
row with that configuration.

Output aligns with the project feature schema used across all datasets:
    latency_ms, jitter_ms, packet_loss_percent, bandwidth_utilization_percent,
    actual_throughput_mbps, flow_duration_sec, packet_count, protocol, application_type,
    link_type, time_of_day, recommended_bandwidth_percent
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "Zenodo_13754300"
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "zenodo_13754300_project_aligned.csv"
)

TESTBEDS = ("ntnu", "wue")

# Raw column name in OWD files; renamed to bw_mhz in aggregation output and tput.
_OWD_GROUP_KEYS = ["gnb", "sdr", "bw", "slots", "ratio", "direction"]
OWD_JOIN_KEYS = ["gnb", "sdr", "bw_mhz", "slots", "ratio", "direction"]

PROJECT_FEATURE_COLUMNS = [
    "latency_ms",
    "jitter_ms",
    "packet_loss_percent",
    "bandwidth_utilization_percent",
    "actual_throughput_mbps",
    "flow_duration_sec",
    "packet_count",
    "protocol",
    "application_type",
    "link_type",
    "time_of_day",
]

# This derived target is kept for compatibility with the older project schema.
# For Zenodo modelling, prefer using actual_throughput_mbps as the first real QoS
# prediction target. recommended_bandwidth_percent is computed from actual
# throughput and offered throughput, so actual_throughput_mbps must be dropped if this
# derived target is ever used for training.
TARGET_COLUMN = "recommended_bandwidth_percent"

TRACEABILITY_COLUMNS = [
    "testbed",
    "gnb",
    "sdr",
    "bw_mhz",
    "slots",
    "ratio",
    "rep",
    "direction",
    "offered_throughput_mbps",
    "source_file",
]


def load_throughput(input_dir: Path) -> pd.DataFrame:
    """Load and concatenate throughput CSVs from both testbeds."""
    frames = []
    for testbed in TESTBEDS:
        tput_file = input_dir / f"{testbed}_tput_all_Throughput.csv"
        if not tput_file.exists():
            print(f"  Warning: throughput file not found: {tput_file.name}")
            continue
        df = pd.read_csv(tput_file, index_col=0)
        df["testbed"] = testbed
        df["source_file"] = tput_file.name
        frames.append(df)
        print(f"  Loaded {len(df):,} rows from {tput_file.name}")

    if not frames:
        raise FileNotFoundError(f"No throughput CSV files found in {input_dir}")

    return pd.concat(frames, ignore_index=True)


def unpivot_uplink_downlink(tput: pd.DataFrame) -> pd.DataFrame:
    """Split each throughput row into separate uplink and downlink rows.

    The raw throughput file stores UL and DL metrics in paired columns on one
    row. Splitting them doubles the sample count and aligns with how OWD files
    store direction as a separate column.
    """
    shared_cols = ["testbed", "source_file", "gnb", "sdr", "bw", "slots", "ratio", "rep", "scenario"]

    uplink = tput[shared_cols].copy()
    uplink["direction"] = "uplink"
    uplink["mbpsactual"] = tput["mbpsactual_uplink"].values
    uplink["meanjitterms"] = tput["meanjitterms_uplink"].values
    uplink["meanloss"] = tput["meanloss_uplink"].values
    uplink["mbpsoffered"] = tput["mbpsoffered_uplink"].values

    downlink = tput[shared_cols].copy()
    downlink["direction"] = "downlink"
    downlink["mbpsactual"] = tput["mbpsactual_downlink"].values
    downlink["meanjitterms"] = tput["meanjitterms_downlink"].values
    downlink["meanloss"] = tput["meanloss_downlink"].values
    downlink["mbpsoffered"] = tput["mbpsoffered_downlink"].values

    result = pd.concat([uplink, downlink], ignore_index=True)
    return result.rename(columns={"bw": "bw_mhz"})


def _init_accum_key(group: pd.DataFrame) -> dict:
    """Initialise a running accumulator entry for one OWD group."""
    first_ts = pd.to_numeric(group["Timestamp"], errors="coerce").dropna()
    return {
        "n": 0,
        "sum_iat": 0.0,
        "sum_sq_iat": 0.0,
        "first_ts": float(first_ts.iloc[0]) if not first_ts.empty else float("nan"),
    }


def aggregate_owd_file(
    owd_file: Path,
    chunksize: int,
) -> pd.DataFrame:
    """Aggregate per-packet OWD data to per-radio-config delay statistics.

    Reads the file in chunks to avoid loading ~12M rows into memory. Accumulates
    running sums for mean and variance computation (Welford-style, but using
    two-pass sums which is fine given floating-point magnitudes here).

    Join key is the radio configuration tuple, not the scenario string, because
    the two file types encode different experiment metadata in their scenario
    fields and cannot be matched on scenario directly.
    """
    accum: dict[tuple, dict] = {}
    rows_read = 0

    for chunk in pd.read_csv(owd_file, index_col=0, chunksize=chunksize):
        # iat == 0 marks the first packet of each burst — no prior arrival to diff.
        chunk = chunk[chunk["iat"] > 0]
        rows_read += len(chunk)

        for key_vals, group in chunk.groupby(_OWD_GROUP_KEYS):
            key = tuple(key_vals) if not isinstance(key_vals, tuple) else key_vals
            iats = group["iat"].to_numpy(dtype=float)

            if key not in accum:
                accum[key] = _init_accum_key(group)

            accum[key]["n"] += len(iats)
            accum[key]["sum_iat"] += float(iats.sum())
            accum[key]["sum_sq_iat"] += float((iats ** 2).sum())

    print(f"    Packets processed (iat > 0): {rows_read:,}")
    print(f"    Unique radio-config/direction groups: {len(accum)}")

    rows = []
    for key, stats in accum.items():
        n = stats["n"]
        if n == 0:
            continue
        mean_iat = stats["sum_iat"] / n
        # Variance via E[x²] - E[x]² — clamped to zero for floating-point noise.
        variance = max(0.0, stats["sum_sq_iat"] / n - mean_iat ** 2)
        std_iat = math.sqrt(variance)
        first_ts = stats["first_ts"]

        if not math.isnan(first_ts):
            time_of_day = pd.Timestamp(first_ts, unit="s").hour
        else:
            time_of_day = -1

        # Map raw OWD group keys to the output schema (bw → bw_mhz).
        row = dict(zip(OWD_JOIN_KEYS, key))
        row.update({
            "owd_latency_ms": mean_iat * 1_000,
            "owd_packet_count": n,
            "owd_flow_duration_sec": mean_iat * n,
            "owd_time_of_day": time_of_day,
        })
        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_owd(input_dir: Path, chunksize: int) -> pd.DataFrame:
    """Aggregate OWD files from both testbeds and tag each row with its testbed."""
    frames = []
    for testbed in TESTBEDS:
        owd_file = input_dir / f"{testbed}_owd_Packets_with_IATs.csv"
        if not owd_file.exists():
            print(f"  Warning: OWD file not found: {owd_file.name}")
            continue
        print(f"  Aggregating {owd_file.name}")
        owd_df = aggregate_owd_file(owd_file, chunksize)
        owd_df["testbed"] = testbed
        frames.append(owd_df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def build_features(
    tput: pd.DataFrame,
    owd: pd.DataFrame | None,
    assumed_link_capacity_mbps: float,
) -> pd.DataFrame:
    """Join throughput and OWD aggregates and map to the project feature schema.

    bandwidth_utilization_percent reflects how hard the link is being pushed
    (offered load vs assumed capacity). recommended_bandwidth_percent reflects
    how much of the offered load was actually delivered by the 5G radio — these
    are intentionally distinct so the target is not identical to an input feature.
    """
    join_keys_with_testbed = ["testbed"] + OWD_JOIN_KEYS

    if owd is not None and not owd.empty:
        df = tput.merge(owd, on=join_keys_with_testbed, how="left")
        latency_ms = df["owd_latency_ms"].fillna(-1)
        packet_count = df["owd_packet_count"].fillna(-1).astype(int)
        flow_duration_sec = df["owd_flow_duration_sec"].fillna(-1)
        time_of_day = df["owd_time_of_day"].fillna(-1).astype(int)
    else:
        df = tput.copy()
        latency_ms = pd.Series(-1.0, index=df.index)
        packet_count = pd.Series(-1, index=df.index)
        flow_duration_sec = pd.Series(-1.0, index=df.index)
        time_of_day = pd.Series(-1, index=df.index)

    bw_util = (df["mbpsoffered"] / assumed_link_capacity_mbps * 100).clip(0, 100)

    # Derived compatibility target:
    #   delivered percent = actual throughput / offered throughput.
    # Do not train a model to predict this while keeping actual_throughput_mbps as an
    # input feature, because that gives the model the answer indirectly.
    recommended = (df["mbpsactual"] / df["mbpsoffered"] * 100).clip(0, 100)

    out = pd.DataFrame(
        {
            "latency_ms": latency_ms,
            "jitter_ms": df["meanjitterms"],
            "packet_loss_percent": (df["meanloss"] * 100).clip(0, 100),
            "bandwidth_utilization_percent": bw_util,
            "actual_throughput_mbps": df["mbpsactual"],
            "flow_duration_sec": flow_duration_sec,
            "packet_count": packet_count,
            "protocol": "UDP",
            "application_type": "throughput_test",
            "link_type": "5G",
            "time_of_day": time_of_day,
            TARGET_COLUMN: recommended,
            "testbed": df["testbed"],
            "gnb": df["gnb"],
            "sdr": df["sdr"],
            "bw_mhz": df["bw_mhz"],
            "slots": df["slots"],
            "ratio": df["ratio"],
            "rep": df["rep"],
            "direction": df["direction"],
            # Offered traffic load is the key controllable input for predicting
            # actual throughput on this dataset.
            "offered_throughput_mbps": df["mbpsoffered"],
            "source_file": df["source_file"],
        }
    )

    return out[PROJECT_FEATURE_COLUMNS + [TARGET_COLUMN] + TRACEABILITY_COLUMNS]


def print_summary(df: pd.DataFrame) -> None:
    """Print a concise summary of the output dataset."""
    print(f"\nRows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    numeric_cols = [
        "latency_ms",
        "jitter_ms",
        "packet_loss_percent",
        "bandwidth_utilization_percent",
        "actual_throughput_mbps",
        TARGET_COLUMN,
    ]
    existing = [c for c in numeric_cols if c in df.columns]
    print("\nFeature summary:")
    print(df[existing].describe().loc[["mean", "min", "max"]].round(3).to_string())

    print(f"\nDirections: {df['direction'].value_counts().to_dict()}")
    print(f"Testbeds:   {df['testbed'].value_counts().to_dict()}")
    print(f"gNB types:  {df['gnb'].value_counts().to_dict()}")
    print("\nModelling note:")
    print(
        "  For Zenodo, prefer actual_throughput_mbps as the first prediction target. "
        "If using recommended_bandwidth_percent, drop actual_throughput_mbps from model inputs "
        "because the target is derived from actual/offered throughput."
    )
    print(f"\nOutput columns:")
    for col in df.columns:
        print(f"  - {col}")


def process_zenodo_dataset(
    input_dir: Path,
    output_path: Path,
    assumed_link_capacity_mbps: float,
    skip_owd: bool,
    chunksize: int,
) -> None:
    """Load, align, and write the Zenodo 5G dataset in project feature format."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading throughput files from: {input_dir}")
    tput = load_throughput(input_dir)
    tput = unpivot_uplink_downlink(tput)
    print(f"Throughput rows after UL/DL unpivot: {len(tput):,}")

    if skip_owd:
        print("\nSkipping OWD aggregation (--skip-owd). Latency, packet count, and flow duration set to -1.")
        owd = None
    else:
        print("\nAggregating OWD packet files (this may take several minutes for ~24M rows)...")
        owd = aggregate_owd(input_dir, chunksize)
        if not owd.empty:
            print(f"OWD aggregation complete. Radio-config groups found: {len(owd):,}")

    print("\nBuilding project-aligned feature table...")
    df = build_features(tput, owd, assumed_link_capacity_mbps)

    df.to_csv(output_path, index=False)
    print_summary(df)
    print(f"\nOutput written to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a project-aligned SD-WAN-style sample from the "
            "Zenodo 5G campus network dataset (record 13754300)."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing the Zenodo CSV files.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path for the output project-aligned CSV.",
    )
    parser.add_argument(
        "--assumed-link-capacity-mbps",
        type=float,
        default=150.0,
        help=(
            "Reference link capacity in Mbps used to compute bandwidth_utilization_percent "
            "(default: 150 Mbps, typical max for 5G NR with 20 MHz downlink)."
        ),
    )
    parser.add_argument(
        "--skip-owd",
        action="store_true",
        help=(
            "Skip OWD file aggregation. Latency, packet_count, and flow_duration_sec "
            "are set to -1. Useful for quick iteration without the ~24M-row read."
        ),
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=100_000,
        help="Rows per chunk when reading OWD files (default: 100,000).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_zenodo_dataset(
        input_dir=args.input_dir,
        output_path=args.output_path,
        assumed_link_capacity_mbps=args.assumed_link_capacity_mbps,
        skip_owd=args.skip_owd,
        chunksize=args.chunksize,
    )
