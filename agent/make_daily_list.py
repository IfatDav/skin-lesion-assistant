"""
make_daily_list.py · build one simulated day's urgent list from PAD-UFES-20 patients
===================================================================================

Use it at the end of the ranking notebook, after the final model has scored the patients.

  scored : a DataFrame with ONE ROW PER PATIENT (after your frozen multi-image rule), with
           the PAD metadata columns and three score columns from your models:
           image_score, symptom_score (raw model outputs); any old final_score ignored

  1. "Arrivals": draw n_arrivals patients at a chosen malignant prevalence (M1 scenario)
  2. Recompute queue-percentile fusion with src.fusion and select the top `slots`
  3. Write daily_cases.csv in the format triage_agent.py reads

The agent never sees the diagnosis or `biopsed`: those columns are not written.
This legacy helper shares descriptive, NON-calibrated rank bands with
build_daily_cases_from_val.py via src/fusion.py. Prefer the root builder
for the reproducible validation-only demo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Notebook 09 imports this module directly from agent/; make the repo root importable.
ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from src.fusion import fuse, rank_band

MALIGNANT = {"MEL", "BCC", "SCC"}                                     # Charter label policy
SYMPTOMS = ("itch", "grew", "hurt", "changed", "bleed", "elevation")  # PAD-UFES-20 columns

def _yes_no(v) -> str:
    s = str(v).strip().lower()
    return "yes" if s in ("true", "1", "yes") else "no" if s in ("false", "0", "no") else "unknown"


def make_daily_list(scored: pd.DataFrame, n_arrivals: int = 100, prevalence: float = 0.10,
                    slots: int = 10, seed: int = 0, out_path: str = "daily_cases.csv") -> pd.DataFrame:
    if not 1 <= slots <= n_arrivals:
        raise ValueError("slots must be between 1 and n_arrivals")
    if scored["patient_id"].duplicated().any():
        raise ValueError("Expected one scored row per patient")
    rng = np.random.default_rng(seed)
    is_mal = scored["diagnostic"].isin(MALIGNANT)
    n_mal = int(round(n_arrivals * prevalence))
    mal = scored[is_mal].sample(n=n_mal, random_state=int(rng.integers(1e9)))
    ben = scored[~is_mal].sample(n=n_arrivals - n_mal, random_state=int(rng.integers(1e9)))
    arrivals = pd.concat([mal, ben], ignore_index=True).set_index("patient_id")
    arrivals["final_score"] = fuse(arrivals["image_score"], arrivals["symptom_score"], w_image=0.6)
    arrivals["image_pct"] = arrivals["image_score"].rank(pct=True)
    arrivals["symptom_pct"] = arrivals["symptom_score"].rank(pct=True)
    day = (arrivals.reset_index()
                   .sort_values(["final_score", "patient_id"], ascending=[False, True])
                   .head(slots))

    rows = []
    for rank, (_, r) in enumerate(day.iterrows(), start=1):
        quality = r.get("image_quality", "not_checked")
        if pd.isna(quality) or not str(quality).strip():
            quality = "not_checked"
        issues = r.get("image_issues", "")
        if pd.isna(issues):
            issues = ""
        row = {
            "patient_id": r["patient_id"],
            "rank": rank,
            "risk_score": round(float(r["final_score"]), 4),
            "risk_band": rank_band(r["final_score"]),
            "image_risk_band": rank_band(r["image_pct"]),
            "symptom_risk_band": rank_band(r["symptom_pct"]),
            "age": int(r["age"]) if pd.notna(r["age"]) else "",
            "body_site": str(r["region"]).lower() if pd.notna(r["region"]) else "",
            "patient_note": "",                                        # PAD has no free text
            "image_path": r["img_id"],
            "image_quality": str(quality),                 # manual check required
            "image_issues": str(issues),                   # issues separated by ;
        }
        row.update({s: _yes_no(r[s]) for s in SYMPTOMS})
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)

    caught = int(day["diagnostic"].isin(MALIGNANT).sum())             # for you, never for the agent
    print(f"{n_arrivals} arrivals, {n_mal} malignant, {slots} slots -> "
          f"{caught} malignant on today's list (sensitivity {caught / max(n_mal, 1):.2f}, "
          f"PPV {caught / slots:.2f}). Saved {out_path}")
    if out["image_quality"].eq("not_checked").any():
        print("WARNING: image_quality not checked; review photos before enabling the agent.")
    return out
