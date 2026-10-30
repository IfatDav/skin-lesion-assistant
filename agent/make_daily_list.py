"""
make_daily_list.py · build one simulated day's urgent list from PAD-UFES-20 patients
===================================================================================

Use it at the end of the ranking notebook, after the final model has scored the patients.

  scored : a DataFrame with ONE ROW PER PATIENT (after your frozen multi-image rule), with
           the PAD metadata columns and three score columns from your models:
           final_score (fusion), image_score, symptom_score (each 0-1, calibrated)

  1. "Arrivals": draw n_arrivals patients at a chosen malignant prevalence (M1 scenario)
  2. Rank them by final_score and keep the top `slots` (the deterministic ranking policy)
  3. Write daily_cases.csv in the format triage_agent.py reads

The agent never sees the diagnosis or `biopsed`: those columns are not written.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MALIGNANT = {"MEL", "BCC", "SCC"}                                     # Charter label policy
SYMPTOMS = ("itch", "grew", "hurt", "changed", "bleed", "elevation")  # PAD-UFES-20 columns

# Score -> band. TODO: freeze these cut-offs on the VALIDATION set (for example score
# quantiles) before running on the test set, and write them in CHARTER.md.
BAND_CUTOFFS = [(0.85, "very high"), (0.60, "high"), (0.30, "moderate"), (0.0, "low")]


def to_band(score: float) -> str:
    return next(band for cut, band in BAND_CUTOFFS if score >= cut)


def _yes_no(v) -> str:
    s = str(v).strip().lower()
    return "yes" if s in ("true", "1", "yes") else "no" if s in ("false", "0", "no") else "unknown"


def make_daily_list(scored: pd.DataFrame, n_arrivals: int = 100, prevalence: float = 0.10,
                    slots: int = 10, seed: int = 0, out_path: str = "daily_cases.csv") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    is_mal = scored["diagnostic"].isin(MALIGNANT)
    n_mal = int(round(n_arrivals * prevalence))
    mal = scored[is_mal].sample(n=n_mal, random_state=int(rng.integers(1e9)))
    ben = scored[~is_mal].sample(n=n_arrivals - n_mal, random_state=int(rng.integers(1e9)))
    day = pd.concat([mal, ben]).sort_values("final_score", ascending=False).head(slots)

    rows = []
    for rank, (_, r) in enumerate(day.iterrows(), start=1):
        row = {
            "patient_id": r["patient_id"],
            "rank": rank,
            "risk_score": round(float(r["final_score"]), 4),
            "risk_band": to_band(r["final_score"]),
            "image_risk_band": to_band(r["image_score"]),
            "symptom_risk_band": to_band(r["symptom_score"]),
            "age": int(r["age"]) if pd.notna(r["age"]) else "",
            "body_site": str(r["region"]).lower() if pd.notna(r["region"]) else "",
            "patient_note": "",                                        # PAD has no free text
            "image_path": r["img_id"],
            "image_quality": r.get("image_quality", "usable"),         # from your quality filter
            "image_issues": r.get("image_issues", ""),                 # issues separated by ;
        }
        row.update({s: _yes_no(r[s]) for s in SYMPTOMS})
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)

    caught = int(day["diagnostic"].isin(MALIGNANT).sum())             # for you, never for the agent
    print(f"{n_arrivals} arrivals, {n_mal} malignant, {slots} slots -> "
          f"{caught} malignant on today's list (sensitivity {caught / max(n_mal, 1):.2f}, "
          f"PPV {caught / slots:.2f}). Saved {out_path}")
    return out
