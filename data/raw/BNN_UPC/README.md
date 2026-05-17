# BNN-UPC Dataset — Local Generation

The pre-built BNN-UPC Challenge 2020 datasets are no longer available for download
from bnn.upc.edu (all links return "Download does not exist" as of May 2026).

This project generates an equivalent dataset using **BNNetSimulator**, the official
OMNeT++-based Docker simulator published by the BNN-UPC research group.

## QoS Class Mapping

| ToS | Class  | Priority | Scheduling weight (Scenario A) |
|-----|--------|----------|-------------------------------|
|  0  | Gold   | Highest  | 60%                           |
|  1  | Silver | Medium   | 30%                           |
|  2  | Bronze | Lowest   | 10%                           |

## How to Generate

### Prerequisites
- Docker Desktop installed and running: https://www.docker.com/products/docker-desktop/
- Python virtual environment active

### Step 1 — Generate input files

```bash
python src/generate_bnnupc_dataset.py \
  --n-topologies 20 \
  --n-tms-per-topology 20 \
  --net-size-min 6 \
  --net-size-max 12
```

This creates `data/raw/BNN_UPC/sim_input/` with:
- `graphs/` — GML topology files (scheduling policy, ToS queues, link capacities)
- `routings/` — shortest-path routing matrices
- `tm/` — traffic matrix files (per-flow: bandwidth, time dist, pkt size dist, ToS)
- `simulation.txt` — manifest listing all (graph, routing, tm) triples
- `conf.yml` — Docker simulator configuration

### Step 2 — Run Docker simulation

```bash
docker run --rm \
  --mount type=bind,src=$(pwd)/data/raw/BNN_UPC/sim_input,dst=/data \
  bnnupc/bnnetsimulator
```

First run downloads the image (~357 MB). Simulation produces results in:
`data/raw/BNN_UPC/sim_input/results/qos_sdwan/`

Each result file contains per-flow: bandwidth, pkts_transmitted, pkts_dropped,
avg_delay, avg_ln_delay, delay percentiles (p10–p90), jitter.

### Step 3 — Process into flat CSV

```bash
python src/process_bnnupc_dataset.py
```

Output: `data/processed/bnnupc_qos_dataset.csv`
One row per src-dst flow, with `qos_class` column (Gold/Silver/Bronze).

### Step 4 — Train PyTorch MLP on log delay

```bash
python src/train_bnnupc_mlp.py
```

The MLP uses the same leakage-safe feature set as the XGBoost log-delay
baseline: identifiers and delay/loss outcome columns are dropped, and the
target is `log_avg_delay`.

Output: `reports/model_results/bnnupc_mlp_log_delay_results.csv`

## References

- BNNetSimulator: https://github.com/BNN-UPC/BNNetSimulator
- Docker Hub image: https://hub.docker.com/r/bnnupc/bnnetsimulator
- datanetAPI: https://github.com/BNN-UPC/datanetAPI/tree/BNNetSimulator
- Original challenge: https://bnn.upc.edu/challenge/gnnet2020/
