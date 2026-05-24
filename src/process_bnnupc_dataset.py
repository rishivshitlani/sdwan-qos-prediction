"""Process BNNetSimulator output into a flat tabular CSV for ML training.

Reads the compressed tar.gz archives produced by BNNetSimulator and flattens
each per-flow simulation result into one CSV row. The output CSV is compatible
with the existing train_baseline.py pipeline.

QoS class mapping:
  ToS=0 → Gold   (highest priority)
  ToS=1 → Silver (medium priority)
  ToS=2 → Bronze (lowest priority)

Output columns
--------------
Identifiers:
  simulation_id, scenario, src_node, dst_node

Input features:
  tos, qos_class, offered_bandwidth, time_distribution,
  routing_hops, n_nodes, max_avg_lambda,
  scheduling_policy, tos_queue_weight, link_bandwidth

Targets (any can be used as the prediction target):
  avg_delay, jitter, packet_loss_rate,
  delay_p10, delay_p50, delay_p90, actual_bandwidth

Usage:
  python src/process_bnnupc_dataset.py [--results-dir ...] [--output-path ...]
"""

from __future__ import annotations

import argparse
import re
import tarfile
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR  = PROJECT_ROOT / "data" / "raw" / "BNN_UPC" / "sim_input" / "results" / "qos_sdwan"
DEFAULT_GRAPHS_DIR   = PROJECT_ROOT / "data" / "raw" / "BNN_UPC" / "sim_input" / "graphs"
DEFAULT_ROUTINGS_DIR = PROJECT_ROOT / "data" / "raw" / "BNN_UPC" / "sim_input" / "routings"
DEFAULT_OUTPUT_PATH  = PROJECT_ROOT / "data" / "processed" / "bnnupc_qos_dataset.csv"

TOS_TO_CLASS = {0: "Gold", 1: "Silver", 2: "Bronze"}
LINK_BANDWIDTH = 100_000  # bits/time-unit (fixed in generate script)

# Fields in each simulationResults path entry (semicolon-separated)
SIM_FIELDS = [
    "actual_bandwidth", "pkts_transmitted", "pkts_dropped",
    "avg_delay", "avg_ln_delay",
    "delay_p10", "delay_p20", "delay_p50", "delay_p80", "delay_p90",
    "jitter",
]


# ---------------------------------------------------------------------------
# Raw text parsers
# ---------------------------------------------------------------------------

def _parse_sim_results_line(line: str) -> tuple[dict, list[dict]]:
    """Parse one line of simulationResults.txt.

    Returns (global_stats, list_of_path_stats).
    Paths with avg_delay == -1 are self-paths and are included as None.
    """
    global_part, paths_part = line.strip().split("|", 1)
    global_vals = global_part.split(",")
    global_stats = {
        "global_packets": float(global_vals[0]),
        "global_losses":  float(global_vals[1]),
        "global_delay":   float(global_vals[2]),
    }

    path_stats = []
    for path_str in paths_part.split(";"):
        path_str = path_str.strip()
        if not path_str:
            continue
        vals = [float(v) for v in path_str.split(",")]
        if len(vals) < len(SIM_FIELDS):
            path_stats.append(None)
            continue
        d = dict(zip(SIM_FIELDS, vals))
        path_stats.append(None if d["avg_delay"] < 0 else d)

    return global_stats, path_stats


def _parse_traffic_line(line: str) -> tuple[float, list[dict | None]]:
    """Parse one line of traffic.txt.

    Returns (max_avg_lambda, list_of_path_traffic).
    Self-paths (first value == -1) are returned as None.
    """
    max_lambda_str, paths_part = line.strip().split("|", 1)
    max_avg_lambda = float(max_lambda_str)

    path_traffic = []
    for path_str in paths_part.split(";"):
        path_str = path_str.strip()
        if not path_str:
            continue
        vals = path_str.split(",")
        # Self-path marker: first value is -1
        if float(vals[0]) < 0:
            path_traffic.append(None)
            continue
        # Last value is always ToS; second value is equivalent_lambda
        tos             = int(float(vals[-1]))
        time_dist       = int(float(vals[0]))
        equiv_lambda    = float(vals[1])
        path_traffic.append({
            "time_distribution":  time_dist,
            "offered_bandwidth":  equiv_lambda,
            "tos":                tos,
        })

    return max_avg_lambda, path_traffic


def _parse_routing_file(routing_path: Path) -> list[list[int]]:
    """Parse a routing file into a list of node paths (one per non-self pair)."""
    paths = []
    with routing_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                paths.append([int(n) for n in line.split(",")])
    return paths


def _topology_stats(graph: nx.Graph, path_nodes: list[int], tos: int) -> dict:
    """Extract per-path topology features from the NetworkX graph.

    Uses the bottleneck (minimum weight) node's policy as the representative.
    For SP nodes the weight is treated as 100% for priority 0, 0% for others.
    """
    policies  = []
    weights   = []

    for node in path_nodes:
        ndata  = graph.nodes[node]
        policy = ndata.get("schedulingPolicy", "WFQ")
        raw_w  = ndata.get("schedulingWeights", "60,30,10")

        if policy == "SP":
            # Strict priority: ToS=0 always wins; encode as 100/0/0
            w_list = [100.0, 0.0, 0.0]
        else:
            w_list = [float(w.strip()) for w in str(raw_w).split(",")]

        # Guard: if fewer weights than ToS, fall back to equal share
        tos_weight = w_list[tos] if tos < len(w_list) else (100.0 / len(w_list))

        policies.append(policy)
        weights.append(tos_weight)

    # Representative policy: most common on path
    dominant_policy = max(set(policies), key=policies.count) if policies else "WFQ"
    min_weight      = float(np.min(weights)) if weights else 33.0
    avg_weight      = float(np.mean(weights)) if weights else 33.0

    return {
        "scheduling_policy": dominant_policy,
        "tos_queue_weight":  avg_weight,
        "min_tos_weight":    min_weight,
    }


# ---------------------------------------------------------------------------
# Archive processor
# ---------------------------------------------------------------------------

def _extract_text(tf: tarfile.TarFile, member_suffix: str) -> str | None:
    """Extract and return text content of the first member ending with suffix."""
    for member in tf.getmembers():
        if member.name.endswith(member_suffix):
            f = tf.extractfile(member)
            if f:
                return f.read().decode("utf-8")
    return None


def process_archive(
    archive_path: Path,
    graphs_dir: Path,
    routings_dir: Path,
) -> list[dict]:
    """Process one results tar.gz archive and return a list of flat row dicts."""
    rows = []

    # Infer scenario from archive filename
    # e.g. results_qos_sdwan_0_9.tar.gz → need graph filename for scenario
    archive_stem = archive_path.stem.replace(".tar", "")

    with tarfile.open(archive_path, "r:gz") as tf:
        sim_results_text = _extract_text(tf, "simulationResults.txt")
        traffic_text     = _extract_text(tf, "traffic.txt")
        input_files_text = _extract_text(tf, "input_files.txt")

    if not all([sim_results_text, traffic_text, input_files_text]):
        print(f"  WARNING: missing files in {archive_path.name}")
        return rows

    sim_lines     = [l for l in sim_results_text.splitlines() if l.strip()]
    traffic_lines = [l for l in traffic_text.splitlines()     if l.strip()]
    input_lines   = [l for l in input_files_text.splitlines() if l.strip()]

    if not (len(sim_lines) == len(traffic_lines) == len(input_lines)):
        print(f"  WARNING: line count mismatch in {archive_path.name}: "
              f"sim={len(sim_lines)} traffic={len(traffic_lines)} input={len(input_lines)}")
        return rows

    # Cache loaded graphs and routing files to avoid re-reading within archive
    graph_cache   : dict[str, nx.Graph]       = {}
    routing_cache : dict[str, list[list[int]]] = {}

    for sim_idx, (sim_line, traffic_line, input_line) in enumerate(
        zip(sim_lines, traffic_lines, input_lines)
    ):
        try:
            # Parse input_files line: sim_num;graph_file;routing_file
            parts        = input_line.strip().split(";")
            sim_num      = int(parts[0])
            graph_name   = parts[1].strip()   # e.g. "graph_A_wfq_fixed_0.txt"
            routing_name = parts[2].strip()

            # Infer scenario from graph filename
            # graph_<scenario>_<topo_idx>.txt  →  e.g. A_wfq_fixed
            m = re.match(r"graph_([^_]+_[^_]+_[^_]+)_\d+\.txt", graph_name)
            scenario = m.group(1) if m else "unknown"

            # Load graph (cached)
            if graph_name not in graph_cache:
                gpath = graphs_dir / graph_name
                if gpath.exists():
                    graph_cache[graph_name] = nx.read_gml(str(gpath), destringizer=int)
                else:
                    graph_cache[graph_name] = None

            # Load routing file (cached)
            # BNNetSimulator strips .txt from routing names in input_files.txt
            if routing_name not in routing_cache:
                rpath = routings_dir / routing_name
                if not rpath.exists():
                    rpath = routings_dir / (routing_name + ".txt")
                if rpath.exists():
                    routing_cache[routing_name] = _parse_routing_file(rpath)
                else:
                    routing_cache[routing_name] = []

            graph   = graph_cache[graph_name]
            routing = routing_cache[routing_name]
            n_nodes = graph.number_of_nodes() if graph is not None else 0

            _, path_stats    = _parse_sim_results_line(sim_line)
            max_avg_lambda, path_traffic = _parse_traffic_line(traffic_line)

            # Build per-flow rows by zipping traffic and results
            # Both lists include self-paths as None; skip them together
            flow_idx = 0  # index into routing list (non-self paths only)

            for traffic_entry, result_entry in zip(path_traffic, path_stats):
                # Skip self-paths
                if traffic_entry is None or result_entry is None:
                    continue

                tos = traffic_entry["tos"]

                # Get routing path for hop count + topology features
                hop_path = routing[flow_idx] if flow_idx < len(routing) else []
                routing_hops = max(1, len(hop_path) - 1)  # edges = nodes - 1

                topo = _topology_stats(graph, hop_path, tos) if (graph and hop_path) else {
                    "scheduling_policy": "unknown",
                    "tos_queue_weight":  33.0,
                    "min_tos_weight":    33.0,
                }

                pkts_tx   = result_entry["pkts_transmitted"]
                pkts_drop = result_entry["pkts_dropped"]
                total_pkts = pkts_tx + pkts_drop
                loss_rate = pkts_drop / total_pkts if total_pkts > 0 else 0.0

                rows.append({
                    # Identifiers
                    "simulation_id":   f"{archive_stem}_{sim_num}",
                    "scenario":        scenario,

                    # Input features
                    "tos":             tos,
                    "qos_class":       TOS_TO_CLASS.get(tos, f"tos_{tos}"),
                    "offered_bandwidth": traffic_entry["offered_bandwidth"],
                    "time_distribution": traffic_entry["time_distribution"],
                    "routing_hops":    routing_hops,
                    "n_nodes":         n_nodes,
                    "max_avg_lambda":  max_avg_lambda,
                    "link_bandwidth":  LINK_BANDWIDTH,
                    **topo,

                    # Targets
                    "avg_delay":       result_entry["avg_delay"],
                    "jitter":          result_entry["jitter"],
                    "packet_loss_rate": loss_rate,
                    "delay_p10":       result_entry["delay_p10"],
                    "delay_p50":       result_entry["delay_p50"],
                    "delay_p90":       result_entry["delay_p90"],
                    "actual_bandwidth": result_entry["actual_bandwidth"],
                })

                flow_idx += 1

        except Exception as exc:
            print(f"  WARNING: error in {archive_path.name} sim {sim_idx}: {exc}")
            continue

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_dataset(
    results_dir: Path,
    graphs_dir: Path,
    routings_dir: Path,
    output_path: Path,
) -> pd.DataFrame:
    archives = sorted(results_dir.glob("*.tar.gz"))
    if not archives:
        raise FileNotFoundError(f"No .tar.gz archives found in {results_dir}")

    print(f"Processing {len(archives)} archives from {results_dir}")

    all_rows: list[dict] = []
    for i, archive in enumerate(archives, 1):
        rows = process_archive(archive, graphs_dir, routings_dir)
        all_rows.extend(rows)
        print(f"  [{i:2d}/{len(archives)}] {archive.name:45s} → {len(rows):5d} flows")

    df = pd.DataFrame(all_rows)

    # Add log-transformed delay so downstream training scripts have the target
    # column they expect without a separate post-processing step.
    df["log_avg_delay"] = np.log(df["avg_delay"].clip(lower=1e-9))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print(f"Total rows : {len(df):,}")
    print(f"Output     : {output_path}")
    print(f"\nColumn summary:")
    print(df.dtypes.to_string())
    print(f"\nQoS class distribution:")
    print(df["qos_class"].value_counts().to_string())
    print(f"\nScheduling policy distribution:")
    print(df["scheduling_policy"].value_counts().to_string())
    print(f"\nTarget statistics (avg_delay):")
    print(df.groupby("qos_class")["avg_delay"].describe().round(6).to_string())
    print(f"{'='*60}\n")

    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flatten BNNetSimulator results into a tabular CSV."
    )
    parser.add_argument("--results-dir",  type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--graphs-dir",   type=Path, default=DEFAULT_GRAPHS_DIR)
    parser.add_argument("--routings-dir", type=Path, default=DEFAULT_ROUTINGS_DIR)
    parser.add_argument("--output-path",  type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_dataset(
        results_dir=args.results_dir,
        graphs_dir=args.graphs_dir,
        routings_dir=args.routings_dir,
        output_path=args.output_path,
    )
