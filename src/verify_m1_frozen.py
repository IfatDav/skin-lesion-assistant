"""Check that the current queue simulator reproduces the frozen M1 report.

Usage: python -m src.verify_m1_frozen

Read-only. Does not change the frozen report or any simulation output.
Requires data/processed/val_scores_fused.csv and the usual Python dependencies.
"""
import json
import math

import pandas as pd

from src.simulate_queue import (
    CAPACITY, OUT_JSON, SCORES, URGENT_SLOTS, simulate, summarize
)


def main():
    frozen = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    pool = pd.read_csv(SCORES)
    expected = frozen["scenarios"]
    actual = [summarize(simulate(pool, triage=False), "arrival order (today)")]

    for slots in (5, 10, 15, 20, 30):
        row = summarize(simulate(pool, urgent_slots=slots), f"{slots} urgent slots per {CAPACITY}")
        row["urgent_slots"] = slots
        actual.append(row)

    if len(actual) != len(expected):
        raise AssertionError(f"Scenario count changed: {len(actual)} != {len(expected)}")

    errors = []
    for old, new in zip(expected, actual):
        if old["scenario"] != new["scenario"]:
            errors.append(f"Scenario name mismatch: {old['scenario']} vs {new['scenario']}")
            continue
        for key, value in old.items():
            if key not in new:
                errors.append(f"{old['scenario']}: missing metric {key}")
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                if not math.isclose(float(value), float(new[key]), rel_tol=1e-12, abs_tol=1e-12):
                    errors.append(f"{old['scenario']}/{key}: frozen={value}, current={new[key]}")
            elif value != new[key]:
                errors.append(f"{old['scenario']}/{key}: frozen={value}, current={new[key]}")

    if errors:
        raise AssertionError("Frozen M1 reproduction FAILED:\n" + "\n".join(errors))
    print(f"PASS: all {len(actual)} frozen M1 scenarios match current simulator outputs.")
    print(f"Frozen file remains unchanged: {OUT_JSON}")


if __name__ == "__main__":
    main()
