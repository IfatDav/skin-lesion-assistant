
"""Reproduce the archived variable-demand M1 stability experiment.

Run: python -m src.m1_stability_variable_demand

This script preserves the 10-run methodology saved at commit c9445c7:
5 seeds, Poisson daily arrivals (mean 100), 110 appointments per day,
with and without 10 priority slots. The original results are archived at
reports/m1_stability_exploratory.json and MUST NOT be overwritten.

Reruns write only to reports/m1_stability_variable_demand_reproduction.json.
This is exploratory simulation, not clinical validation.
"""

from pathlib import Path
import json

import pandas as pd

from src.simulate_queue import (
    simulate,
    summarize,
    SCORES,
)

OUTPUT = Path(__file__).resolve().parents[1] / "reports" / "m1_stability_variable_demand_reproduction.json"

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


def main():
    results = run_stability()

    report = {
        "status": "exploratory - multi-seed stability analysis",
        "seeds": SEEDS,
        "assumptions": {
            "capacity_per_day": 110,
            "mean_arrivals_per_day": 100,
            "variable_arrivals": True,
            "priority_slots_with_triage": 10,
            "comparison": "with triage vs without triage"
        },
        "limitations": [
            "Same seeds do not guarantee identical patient arrivals across policies.",
            "Results are simulation estimates, not clinical validation."
        ],
        "results": results
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print(f"Stability results saved to: {OUTPUT}")


if __name__ == "__main__":
    main()


