
"""Exploratory M1 stability analysis.

Runs multiple random seeds without modifying
the frozen M1 evaluation or its original report.
"""

from pathlib import Path
import json

import pandas as pd

from src.simulate_queue import (
    simulate,
    summarize,
    SCORES,
)

OUTPUT = Path("reports/m1_stability_exploratory.json")

SEEDS = [42, 123, 456, 789, 2026]


def run_stability():
    pool = pd.read_csv(SCORES)
    results = []

    for seed in SEEDS:
        for triage in (False, True):
            seen = simulate(
                pool,
                capacity=110,
                urgent_slots=10 if triage else 0,
                arrivals_per_day=100,
                variable_arrivals=True,
                triage=triage,
                seed=seed,
            )

            result = summarize(
                seen,
                "with triage" if triage else "without triage"
            )
            result["seed"] = seed
            results.append(result)

    return results

