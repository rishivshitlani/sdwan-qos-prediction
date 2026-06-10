"""Single source of truth for project-wide constants and paths.

Every value here used to be copy-pasted across several scripts in ``src/``.
Centralising them prevents silent drift — especially for
``BNNUPC_LEAKAGE_COLUMNS``, where a stale copy would let a model train on
outcome columns and invalidate its results without any error.
"""

from __future__ import annotations

from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# This file lives at src/sdwan_qos/config.py, so the project root is two
# levels up. Valid for the editable/in-repo layout used by this project.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports" / "model_results"

BNNUPC_PROCESSED_DATASET = PROCESSED_DATA_DIR / "bnnupc_qos_dataset.csv"


# ---------------------------------------------------------------------------
# QoS class taxonomy
# ---------------------------------------------------------------------------

QOS_ORDER = ["Gold", "Silver", "Bronze"]

# BNNetSimulator ToS value → project QoS class.
TOS_TO_CLASS = {0: "Gold", 1: "Silver", 2: "Bronze"}
CLASS_TO_TOS = {qos_class: tos for tos, qos_class in TOS_TO_CLASS.items()}


# ---------------------------------------------------------------------------
# BNN-UPC dataset schema and leakage policy
# ---------------------------------------------------------------------------

# Fixed link capacity (bits/time-unit) used both when generating BNNetSimulator
# topologies and when flattening its results. Must be one value, not two.
LINK_BANDWIDTH = 100_000

# Trace-only identifiers: never model inputs.
BNNUPC_IDENTIFIER_COLUMNS = [
    "simulation_id",
    "scenario",
]

# Measured outcome columns: never inputs when predicting any QoS target,
# because they are only known after the traffic has been observed.
BNNUPC_OUTCOME_COLUMNS = [
    "avg_delay",
    "jitter",
    "packet_loss_rate",
    "delay_p10",
    "delay_p50",
    "delay_p90",
    "actual_bandwidth",
]

# The standard drop list for the log-delay experiments. The target column
# (log_avg_delay) is removed from X separately by each loader.
BNNUPC_LEAKAGE_COLUMNS = [*BNNUPC_IDENTIFIER_COLUMNS, *BNNUPC_OUTCOME_COLUMNS]


# ---------------------------------------------------------------------------
# SLA thresholds (milliseconds)
# ---------------------------------------------------------------------------
# Two deliberately different threshold sets exist in this project. They were
# previously both named DEFAULT_SLA_MS in different modules, which invited
# mix-ups; the distinct names below are intentional.

# Evaluation thresholds for SLA-violation trigger metrics. Bronze was tuned
# from 100 ms to 60 ms (reproduce with evaluate_bnnupc_qos_slices.py
# --bronze-sweep): improves recall 0.368 -> 0.568 while staying strictly more
# lenient than Silver (50 ms).
EVAL_SLA_MS = {
    "Gold": 30.0,
    "Silver": 50.0,
    "Bronze": 60.0,
}

# Allocation targets used by the WFQ bandwidth recommender. These are the
# delays each class *should* meet, not the trigger thresholds above.
ALLOCATION_SLA_MS = {
    "Gold": 20.0,
    "Silver": 50.0,
    "Bronze": 100.0,
}

# Penalty weights for SLA violations when ranking allocation profiles.
SLA_PENALTY_WEIGHTS = {
    "Gold": 3.0,
    "Silver": 2.0,
    "Bronze": 1.0,
}
