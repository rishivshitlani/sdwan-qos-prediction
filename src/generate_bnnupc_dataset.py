"""Generate a BNN-UPC-format QoS dataset using BNNetSimulator.

This script produces the input files (topology GML, routing, traffic matrices,
simulation manifest, and Docker config) that BNNetSimulator consumes. Running
the generated Docker command then simulates the network and writes per-flow
delay/jitter/loss results in the same format as the original BNN-UPC Challenge
2020 datasets.

QoS class mapping used throughout:
  ToS 0 → Gold   (real-time, highest priority)
  ToS 1 → Silver (business/interactive, medium priority)
  ToS 2 → Bronze (bulk/best-effort, lowest priority)

Four scheduling scenarios are generated (25% of samples each), mirroring the
original ch20 dataset structure:
  Scenario A: WFQ, fixed weights  60/30/10 for Gold/Silver/Bronze
  Scenario B: WFQ, 5 random weight profiles per node
  Scenario C: Mixed SP/WFQ/DRR per node, 5 weight profiles each
  Scenario D: Same as C, but ToS assigned equiprobably (33% each)

Usage (after installing requirements):
  python src/generate_bnnupc_dataset.py [--output-dir data/raw/BNN_UPC/sim_input]
                                        [--n-topologies 5]
                                        [--n-tms-per-topology 20]
                                        [--net-size-min 6]
                                        [--net-size-max 10]
                                        [--threads 4]

After generation, run the Docker container (printed at the end of this script):
  docker run --rm \\
    --mount type=bind,src=<abs_path_to_output_dir>,dst=/data \\
    bnnupc/bnnetsimulator
"""

from __future__ import annotations

import argparse
import os
import random
import re
from pathlib import Path
from typing import NamedTuple

import networkx as nx
import yaml


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "BNN_UPC" / "sim_input"
DATASET_NAME = "qos_sdwan"

# ToS → QoS class label (used only in comments / post-processing)
TOS_TO_CLASS = {0: "Gold", 1: "Silver", 2: "Bronze"}

# WFQ weight profiles for each scheduling scenario (Gold/Silver/Bronze %)
WFQ_PROFILES = [
    "90,5,5",
    "33,33,34",
    "60,30,10",
    "50,40,10",
    "75,20,5",
]

DRR_PROFILES = [
    "80,10,10",
    "33,33,34",
    "60,30,10",
    "70,20,10",
    "65,25,10",
]

# Traffic intensity range (bits/time-unit).
# Raised from ch20 defaults (400-2000) to create meaningful congestion so that
# QoS scheduling weights (Gold=60%, Bronze=10%) produce visible delay/loss differences.
# At 2000 bit/tu the network is ~14% utilised and all queues are empty — no differentiation.
# At 10000 bit/tu small topologies (6-8 nodes) reach 50-80% utilisation where scheduling matters.
MAX_AVG_LAMBDA = 10_000
MIN_AVG_LAMBDA = 2_000

# ToS probability distributions per scenario (matches original ch20)
TOS_PROBS_SKEWED = [0.10, 0.30, 0.60]   # Scenarios A/B/C: 10% Gold, 30% Silver, 60% Bronze
TOS_PROBS_EQUAL  = [0.333, 0.333, 0.334] # Scenario D: equiprobable

# Packet size distributions (format used by BNNetSimulator generic dist)
PKT_DIST_SMALL = "0,300,0.5,1700,0.5"     # generic: 300-bit or 1700-bit pkts
PKT_DIST_LARGE = "0,500,0.4,1000,0.3,1400,0.3"  # generic: 500/1000/1400-bit

LINK_BANDWIDTH = 100_000  # bits/time-unit (matches ch20)
BUFFER_SIZE    = 32_000   # bits


# ---------------------------------------------------------------------------
# Scenario definition
# ---------------------------------------------------------------------------

class Scenario(NamedTuple):
    name: str
    scheduling_policy: str   # "WFQ", "SP", "DRR", or "MIXED"
    tos_probs: list[float]


SCENARIOS = [
    Scenario("A_wfq_fixed",   "WFQ",   TOS_PROBS_SKEWED),
    Scenario("B_wfq_profiles","WFQ",   TOS_PROBS_SKEWED),
    Scenario("C_mixed_policy","MIXED",  TOS_PROBS_SKEWED),
    Scenario("D_mixed_equal", "MIXED",  TOS_PROBS_EQUAL),
]


# ---------------------------------------------------------------------------
# Topology generation
# ---------------------------------------------------------------------------

def _random_connected_graph(net_size: int, rng: random.Random) -> nx.Graph:
    """Generate a random connected graph with net_size nodes."""
    while True:
        G = nx.Graph()
        degree_budget = [
            rng.choices([2, 3, 4, 5, 6], weights=[0.34, 0.35, 0.20, 0.10, 0.01])[0]
            for _ in range(net_size)
        ]
        remaining = list(range(net_size))
        for n in range(net_size):
            G.add_node(n)

        while len(remaining) > 1:
            n0 = rng.choice(remaining)
            candidates = [n for n in remaining if n != n0 and not G.has_edge(n0, n)]
            if not candidates:
                remaining.remove(n0)
                continue
            n1 = rng.choice(candidates)
            G.add_edge(n0, n1, bandwidth=LINK_BANDWIDTH)
            for n in (n0, n1):
                degree_budget[n] -= 1
                if degree_budget[n] <= 0 and n in remaining:
                    remaining.remove(n)

        if nx.is_connected(G):
            return G


def generate_topology(
    net_size: int,
    graph_file: Path,
    scenario: Scenario,
    rng: random.Random,
) -> nx.Graph:
    """Write a GML topology file and return the NetworkX graph.

    Each node gets:
    - levelsToS = 3 (Gold=0, Silver=1, Bronze=2)
    - 3 QoS queues (one per ToS class)
    - Scheduling policy from the scenario
    - Random weight profile (for WFQ/DRR) or fixed profile
    """
    G = _random_connected_graph(net_size, rng)
    G.graph["levelsToS"] = 3

    for n in G.nodes:
        policy = _pick_policy(scenario, rng)
        weights = _pick_weights(policy, scenario, rng)
        G.nodes[n]["schedulingPolicy"] = policy
        G.nodes[n]["tosToQoSqueue"] = "0;1;2"   # each ToS → its own queue
        G.nodes[n]["schedulingWeights"] = weights
        G.nodes[n]["bufferSizes"] = BUFFER_SIZE

    nx.write_gml(G, str(graph_file))
    return G


def _pick_policy(scenario: Scenario, rng: random.Random) -> str:
    if scenario.scheduling_policy == "MIXED":
        return rng.choice(["SP", "WFQ", "DRR"])
    return scenario.scheduling_policy


def _pick_weights(policy: str, scenario: Scenario, rng: random.Random) -> str:
    if policy == "SP":
        return "1,1,1"  # SP ignores weights; placeholder required by simulator
    profiles = WFQ_PROFILES if policy == "WFQ" else DRR_PROFILES
    if scenario.name == "A_wfq_fixed":
        return "60,30,10"  # Scenario A: fixed weights from ch20 Scenario 1
    return rng.choice(profiles)


# ---------------------------------------------------------------------------
# Routing generation
# ---------------------------------------------------------------------------

def generate_routing(G: nx.Graph, routing_file: Path) -> None:
    """Write destination-based RIB matrix (one path per line, nodes comma-separated)."""
    paths = dict(nx.shortest_path(G))
    with routing_file.open("w") as fd:
        for src in G:
            for dst in G:
                if src == dst:
                    continue
                path = ",".join(str(n) for n in paths[src][dst])
                fd.write(path + "\n")


# ---------------------------------------------------------------------------
# Traffic matrix generation
# ---------------------------------------------------------------------------

def generate_tm(
    G: nx.Graph,
    tm_file: Path,
    tos_probs: list[float],
    rng: random.Random,
) -> None:
    """Write a traffic matrix file.

    Format per line:
      src,dst,avg_bw,time_distribution,pkt_size_distribution,ToS

    Time distributions used:
      0 = Poisson/Exponential
      1 = CBR (Constant Bit Rate)
      2,10,5 = ON-OFF (avg on=10, avg off=5)

    Packet size distributions (generic format):
      0,<size1>,<prob1>,<size2>,<prob2>,...
    """
    time_dists = ["0", "1", "2,10,5"]
    pkt_dists  = [PKT_DIST_SMALL, PKT_DIST_LARGE]

    with tm_file.open("w") as fd:
        for src in G:
            for dst in G:
                if src == dst:
                    continue
                avg_bw = rng.randint(MIN_AVG_LAMBDA, MAX_AVG_LAMBDA)
                td     = rng.choice(time_dists)
                sd     = rng.choice(pkt_dists)
                tos    = rng.choices([0, 1, 2], weights=tos_probs)[0]
                fd.write(f"{src},{dst},{avg_bw},{td},{sd},{tos}\n")


# ---------------------------------------------------------------------------
# Simulation manifest + Docker config
# ---------------------------------------------------------------------------

def write_simulation_manifest(
    output_dir: Path,
    entries: list[tuple[str, str, str]],
) -> Path:
    """Write simulation.txt listing all (graph, routing, tm) triples."""
    sim_file = output_dir / "simulation.txt"
    with sim_file.open("w") as fd:
        for graph_rel, routing_rel, tm_rel in entries:
            # Normalise to forward slashes (Docker runs on Linux)
            line = f"{graph_rel},{routing_rel},{tm_rel}\n"
            fd.write(line.replace("\\", "/"))
    return sim_file


def write_docker_config(output_dir: Path, threads: int) -> Path:
    """Write conf.yml consumed by BNNetSimulator."""
    conf = {
        "threads":         threads,
        "dataset_name":    DATASET_NAME,
        "samples_per_file": 10,
        "rm_prev_results": "n",
        "write_pkt_info":  "n",
    }
    conf_file = output_dir / "conf.yml"
    with conf_file.open("w") as fd:
        yaml.dump(conf, fd)
    return conf_file


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

def generate_dataset(
    output_dir: Path,
    n_topologies: int,
    n_tms_per_topology: int,
    net_size_min: int,
    net_size_max: int,
    threads: int,
    random_seed: int,
) -> None:
    rng = random.Random(random_seed)

    graphs_dir   = output_dir / "graphs"
    routings_dir = output_dir / "routings"
    tms_dir      = output_dir / "tm"
    for d in (graphs_dir, routings_dir, tms_dir):
        d.mkdir(parents=True, exist_ok=True)

    sim_entries: list[tuple[str, str, str]] = []

    n_sizes = net_size_max - net_size_min + 1
    sizes = [net_size_min + (i % n_sizes) for i in range(n_topologies)]

    # Distribute samples evenly across 4 scenarios
    scenarios_cycle = SCENARIOS * (n_topologies // len(SCENARIOS) + 1)

    for topo_idx, (net_size, scenario) in enumerate(zip(sizes, scenarios_cycle)):
        graph_name   = f"graph_{scenario.name}_{topo_idx}.txt"
        routing_name = f"routing_{scenario.name}_{topo_idx}.txt"

        graph_file   = graphs_dir   / graph_name
        routing_file = routings_dir / routing_name

        G = generate_topology(net_size, graph_file, scenario, rng)
        generate_routing(G, routing_file)

        for tm_idx in range(n_tms_per_topology):
            tm_name = f"tm_{scenario.name}_{topo_idx}_{tm_idx}.txt"
            tm_file = tms_dir / tm_name
            generate_tm(G, tm_file, scenario.tos_probs, rng)

            # Paths in simulation.txt are relative to output_dir
            sim_entries.append((
                f"graphs/{graph_name}",
                f"routings/{routing_name}",
                f"tm/{tm_name}",
            ))

        print(
            f"  [{topo_idx+1:3d}/{n_topologies}] "
            f"scenario={scenario.name:20s} "
            f"nodes={net_size:2d} "
            f"tms={n_tms_per_topology}"
        )

    manifest = write_simulation_manifest(output_dir, sim_entries)
    conf     = write_docker_config(output_dir, threads)

    total_samples = len(sim_entries)
    abs_output    = output_dir.resolve()

    print(f"\n{'='*60}")
    print(f"Generated {total_samples} simulation samples")
    print(f"  Manifest : {manifest}")
    print(f"  Config   : {conf}")
    print(f"\nScenario breakdown (~{total_samples // 4} samples each):")
    for sc in SCENARIOS:
        print(f"  {sc.name:22s}  policy={sc.scheduling_policy:5s}  "
              f"ToS probs={[round(p,2) for p in sc.tos_probs]}")
    print(f"\nQoS class mapping:")
    for tos, cls in TOS_TO_CLASS.items():
        print(f"  ToS={tos} → {cls}")
    print(f"\n{'='*60}")
    print("Step 2 — run Docker simulation:")
    print(f"\n  docker run --rm \\")
    print(f"    --mount type=bind,src={abs_output},dst=/data \\")
    print(f"    bnnupc/bnnetsimulator")
    print(f"\nResults will appear in:")
    print(f"  {abs_output}/results/{DATASET_NAME}/")
    print(f"\nStep 3 — process results:")
    print(f"  python src/process_bnnupc_dataset.py")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate BNNetSimulator input files for QoS-class-aware SD-WAN experiments."
    )
    parser.add_argument("--output-dir",  type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-topologies",         type=int, default=20,
                        help="Number of distinct topologies to generate")
    parser.add_argument("--n-tms-per-topology",   type=int, default=20,
                        help="Traffic matrices per topology (= samples per topology)")
    parser.add_argument("--net-size-min",         type=int, default=6)
    parser.add_argument("--net-size-max",         type=int, default=12)
    parser.add_argument("--threads",              type=int, default=4,
                        help="Docker simulation threads")
    parser.add_argument("--random-seed",          type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Generating BNN-UPC style dataset")
    print(f"  Output dir  : {args.output_dir}")
    print(f"  Topologies  : {args.n_topologies}")
    print(f"  TMs/topology: {args.n_tms_per_topology}")
    print(f"  Net sizes   : {args.net_size_min}–{args.net_size_max} nodes")
    print(f"  Total samples: {args.n_topologies * args.n_tms_per_topology}")
    print()
    generate_dataset(
        output_dir=args.output_dir,
        n_topologies=args.n_topologies,
        n_tms_per_topology=args.n_tms_per_topology,
        net_size_min=args.net_size_min,
        net_size_max=args.net_size_max,
        threads=args.threads,
        random_seed=args.random_seed,
    )
