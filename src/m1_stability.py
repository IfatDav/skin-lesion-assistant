

"""Exploratory M1 stability analysis.

    python -m src.m1_stability

Five seeds per configuration, with arrivals matched to capacity so the waiting list holds steady and no
initial backlog drains. The two configurations differ only in the ratio of priority slots to expected
malignant arrivals per day, which is the quantity the frozen M1 median turns out to depend on.

Writes reports/m1_stability_balanced_demand_exploratory.json. The frozen M1 report is never read or written here.
"""
import json
from pathlib import Path

import pandas as pd

from src.simulate_queue import PREVALENCE, SCORES, URGENT_SLOTS, simulate, summarize

try:
    from src import config as C
    OUTPUT = C.REPO_ROOT / "reports" / "m1_stability_balanced_demand_exploratory.json"
except ImportError:                                    # runnable as a plain script too
    OUTPUT = Path(__file__).resolve().parents[1] / "reports" / "m1_stability_balanced_demand_exploratory.json"

SEEDS = [42, 123, 456, 789, 2026]

# (arrivals per day, appointments per day). Arrivals equal capacity in both, so the list is in steady
# state and neither arm is contaminated by a draining backlog.
#   100 arrivals x 10% prevalence = 10 malignant a day against 10 priority slots -> ratio 1.00
#   110 arrivals x 10% prevalence = 11 malignant a day against 10 priority slots -> ratio 0.91
CONFIGS = [(100, 100), (110, 110)]


def run_stability():
    pool = pd.read_csv(SCORES)
    results = []
    for arrivals, capacity in CONFIGS:
        for seed in SEEDS:
            for triage in (False, True):
                slots = URGENT_SLOTS if triage else 0
                seen = simulate(pool, capacity=capacity, urgent_slots=slots,
                                arrivals_per_day=arrivals, variable_arrivals=False,
                                triage=triage, seed=seed)
                label = (f"{arrivals} arrivals / {capacity} slots, "
                         + (f"{slots} priority" if triage else "no triage"))
                row = summarize(seen, label)
                row["seed"] = seed
                row["arrivals_per_day"] = arrivals
                row["capacity_per_day"] = capacity
                row["priority_slots"] = slots
                row["slots_per_malignant_arrival"] = (
                    round(slots / (arrivals * PREVALENCE), 2) if triage else None
                )
                results.append(row)
    return results


def main():
    results = run_stability()

    report = {
        "status": "exploratory - multi-seed stability analysis, not a replacement for frozen M1",
        "seeds": SEEDS,
        "assumptions": {
            "configurations": [{"arrivals_per_day": a, "capacity_per_day": c} for a, c in CONFIGS],
            "arrivals_matched_to_capacity": True,
            "variable_arrivals": False,
            "malignancy_prevalence": PREVALENCE,
            "priority_slots_with_triage": URGENT_SLOTS,
            "comparison": "with triage vs without triage, paired per seed",
            "note": "per-scenario arrivals, capacity and slot ratio are recorded on each result row",
        },
        "limitations": [
            "The arrival stream is identical across policies for a given seed: the triage branch "
            "consumes no randomness, so with/without triage is a paired comparison on the same "
            "simulated patients.",
            "Five seeds bound run-to-run variation; they do not establish individual-level safety.",
            "Results are simulation estimates, not clinical validation.",
        ],
        "results": results,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"{len(results)} runs saved to: {OUTPUT}")
    for r in results:
        print(f"  seed {r['seed']:>5} | {r['scenario']:<44} | "
              f"median {r['median_days_malignant']:>5.1f} | "
              f">7d {r['share_malignant_over_target']:>6.1%} | "
              f"in slot {r['share_malignant_in_urgent_slot']:>6.1%}")


if __name__ == "__main__":
    main()
