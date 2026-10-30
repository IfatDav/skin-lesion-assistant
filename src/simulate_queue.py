"""M1 · Days to appointment, simulated on a dermatology waiting list.

    python -m src.simulate_queue

What it does. A clinic sees `capacity` patients a day and the same number arrive, so the waiting list
holds steady. Of the day's appointments, `urgent_slots` are kept for priority review and the rest are
given out in arrival order. Every morning the whole waiting list is re-ranked by the final model --
60% image rank + 40% questionnaire rank, computed within that morning's list, exactly as `src/triage.py`
does -- and the urgent slots go to the top of that ranking. Everyone else keeps their place in the queue.

The patients are drawn by resampling the 220 PAD validation patients with their real model scores,
reweighted to the stated malignancy prevalence. No test patient is touched.

The baseline is today's practice: the same clinic with no urgent slots, every patient in arrival order.
The charter anchors it at a median of 24.5 days for an Israeli public dermatology appointment, so the
initial queue is set to `24.5 x capacity` and the baseline reproduces that median by construction. The
charter also says queue metrics are scenario results, not predictions, so the sweep matters more than
any single row.

M1 asks for the median wait of patients with a malignant lesion, against that baseline. The median alone
flatters the policy: a patient the model does not rank highly gets no benefit at all, so the share of
cancer patients still waiting longer than the target is reported beside it.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src import config as C
    SCORES = C.REPO_ROOT / "data" / "processed" / "val_scores_fused.csv"
    OUT_JSON = C.REPO_ROOT / "reports" / "m1_queue_simulation.json"
except ImportError:                                    # runnable as a plain script too
    ROOT = Path(__file__).resolve().parents[1]
    SCORES = ROOT / "data" / "processed" / "val_scores_fused.csv"
    OUT_JSON = ROOT / "reports" / "m1_queue_simulation.json"

BASELINE_MEDIAN_DAYS = 24.5        # charter section 1, Israeli public dermatology
TARGET_DAYS = 7                    # charter M1
W_IMAGE = 0.6                      # models/final_model.json
SEED = 42

CAPACITY = 100                     # appointments a day; arrivals match, so the list holds steady
PREVALENCE = 0.10                  # share of arrivals with a malignant lesion
URGENT_SLOTS = 10                  # of the day's appointments, kept for priority review
DAYS = 400
BURN_IN = 120                      # days discarded before measuring, so the list is in steady state


def draw(rng, n, pool, prevalence):
    """n patients resampled from the validation pool, reweighted to the stated prevalence."""
    pos, neg = pool[pool["y"] == 1], pool[pool["y"] == 0]
    n_pos = rng.binomial(n, prevalence)
    idx = np.concatenate([rng.choice(pos.index, n_pos, replace=True),
                          rng.choice(neg.index, n - n_pos, replace=True)])
    return pool.loc[idx].reset_index(drop=True)


def simulate(pool, capacity=CAPACITY, urgent_slots=URGENT_SLOTS, prevalence=PREVALENCE,
             days=DAYS, burn_in=BURN_IN, seed=SEED, triage=True):
    """Returns one row per patient seen after the burn-in: their wait in days and their label."""
    rng = np.random.default_rng(seed)
    queue = draw(rng, int(round(BASELINE_MEDIAN_DAYS * capacity)), pool, prevalence)
    queue["arrival_day"] = rng.integers(-int(round(BASELINE_MEDIAN_DAYS * 2)), 1, len(queue))
    seen = []

    for day in range(days):
        # The morning ranking runs on the list as it stood overnight, so a photo submitted today is
        # ranked tomorrow at the earliest. Nobody is seen on the day they arrive.
        eligible = queue.index[queue["arrival_day"] < day]

        taken = []
        if triage and urgent_slots > 0:
            # re-ranked within this morning's list, as src/triage.py does for a day's queue
            waiting = queue.loc[eligible]
            score = (W_IMAGE * waiting["image_score"].rank(pct=True)
                     + (1 - W_IMAGE) * waiting["questionnaire_score"].rank(pct=True))
            taken += list(score.nlargest(min(urgent_slots, len(score))).index)

        routine = capacity - len(taken)
        if routine > 0:
            rest = queue.loc[eligible].drop(index=taken)
            taken += list(rest.nsmallest(min(routine, len(rest)), "arrival_day").index)

        served = queue.loc[taken].assign(seen_day=day)
        served["urgent"] = [i in set(taken[:urgent_slots]) if triage else False for i in served.index]
        if day >= burn_in:
            seen.append(served)

        arrivals = draw(rng, capacity, pool, prevalence)
        arrivals["arrival_day"] = day
        queue = pd.concat([queue.drop(index=taken), arrivals], ignore_index=True)

    out = pd.concat(seen, ignore_index=True)
    out["wait_days"] = out["seen_day"] - out["arrival_day"]
    return out


def summarize(seen, label):
    m = seen[seen["y"] == 1]
    mal, ben = m["wait_days"], seen[seen["y"] == 0]["wait_days"]
    return {"scenario": label,
            "n_malignant_seen": int(len(mal)),
            "median_days_malignant": float(mal.median()),
            "p75_days_malignant": float(mal.quantile(0.75)),
            "p90_days_malignant": float(mal.quantile(0.90)),
            "share_malignant_over_target": float((mal > TARGET_DAYS).mean()),
            "share_malignant_in_urgent_slot": float(m["urgent"].mean()),
            "median_days_benign": float(ben.median())}


def main():
    pool = pd.read_csv(SCORES)
    print(f"pool: {len(pool)} validation patients, {int(pool['y'].sum())} malignant\n")

    base = summarize(simulate(pool, triage=False), "arrival order (today)")
    print(f"BASELINE  median {base['median_days_malignant']:.1f} days for malignant patients "
          f"(the charter anchors today's practice at {BASELINE_MEDIAN_DAYS})\n")

    rows = [base]
    print("  urgent slots | reached a slot |  median   p75    p90  | still > 7 days | benign median")
    for slots in (5, 10, 15, 20, 30):
        r = summarize(simulate(pool, urgent_slots=slots), f"{slots} urgent slots per {CAPACITY}")
        r["urgent_slots"] = slots
        rows.append(r)
        print(f"  {slots:>5} per {CAPACITY}  | {r['share_malignant_in_urgent_slot']:>13.0%} "
              f"| {r['median_days_malignant']:>6.1f} {r['p75_days_malignant']:>6.1f} "
              f"{r['p90_days_malignant']:>6.1f}  | {r['share_malignant_over_target']:>13.0%} "
              f"| {r['median_days_benign']:>8.1f}")

    primary = next(r for r in rows if r.get("urgent_slots") == URGENT_SLOTS)
    cut = base["median_days_malignant"] - primary["median_days_malignant"]
    print(f"\nPRIMARY SCENARIO ({URGENT_SLOTS} urgent slots per {CAPACITY}, "
          f"{PREVALENCE:.0%} prevalence)")
    print(f"  median wait for a malignant lesion: {base['median_days_malignant']:.1f} -> "
          f"{primary['median_days_malignant']:.1f} days "
          f"({cut:.1f} days, {cut / base['median_days_malignant']:.0%} shorter)")
    print(f"  M1 target <= {TARGET_DAYS} days: "
          f"{'MET' if primary['median_days_malignant'] <= TARGET_DAYS else 'NOT MET'}")
    print(f"  but {primary['share_malignant_over_target']:.0%} of malignant patients still wait "
          f"longer than {TARGET_DAYS} days: the model did not rank them into an urgent slot.")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(
        {"assumptions": {"capacity_per_day": CAPACITY, "arrivals_per_day": CAPACITY,
                         "prevalence": PREVALENCE, "fusion_w_image": W_IMAGE,
                         "baseline_median_days": BASELINE_MEDIAN_DAYS, "target_days": TARGET_DAYS,
                         "days_simulated": DAYS, "burn_in_days": BURN_IN, "seed": SEED,
                         "patients": "the 220 PAD validation patients, resampled to the prevalence"},
         "scenarios": rows}, indent=1))
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
