"""The charter's operating point, implemented exactly as in notebook 02 (Cell 8b) so results are comparable.

A daily queue of 100 arrivals with a fixed number of urgent slots; patients are reweighted to a stated
malignant prevalence, the top-scoring share up to the slot budget is flagged.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

DAILY_ARRIVALS = 100
SLOT_SCENARIOS = [5, 10, 15]
PREVALENCES = [0.05, 0.10, 0.20]
PRIMARY = (10, 0.10)  # slots per day, prevalence


def to_patients(patient_id, y, score) -> pd.DataFrame:
    """Frozen aggregation rule: image -> lesion -> patient, max at each step (equivalent to max over images)."""
    return (pd.DataFrame({"patient_id": patient_id, "y": y, "score": score})
              .groupby("patient_id").agg(y=("y", "max"), score=("score", "max")))


def at_budget(y, s, prevalence, budget):
    y, s = np.asarray(y), np.asarray(s)
    n_pos = y.sum()
    w = np.where(y == 1, prevalence / n_pos, (1 - prevalence) / (len(y) - n_pos))
    order = np.argsort(-s, kind="stable")
    w_o, y_o = w[order], y[order]
    flagged = np.cumsum(w_o) <= budget + 1e-12
    sens = w_o[flagged & (y_o == 1)].sum() / prevalence
    flag_rate = w_o[flagged].sum()
    ppv = prevalence * sens / flag_rate if flag_rate > 0 else np.nan
    return sens, ppv, flag_rate


def flagged_mask(y, s, prevalence, budget) -> np.ndarray:
    """Which patients at_budget flags, in the original order."""
    y, s = np.asarray(y), np.asarray(s)
    n_pos = y.sum()
    w = np.where(y == 1, prevalence / n_pos, (1 - prevalence) / (len(y) - n_pos))
    order = np.argsort(-s, kind="stable")
    mask = np.zeros(len(y), dtype=bool)
    mask[order] = np.cumsum(w[order]) <= budget + 1e-12
    return mask


def boot_ci(y, s, metric, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    vals = [metric(y[idx], s[idx]) for idx in
            (np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]) for _ in range(n))]
    return np.percentile(vals, [2.5, 97.5])


def report(pat: pd.DataFrame, n_boot: int = 2000) -> dict:
    """Patient-level AUC with CI and the 9 slot x prevalence scenarios."""
    lo, hi = boot_ci(pat["y"], pat["score"], roc_auc_score, n=n_boot)
    out = {"n_patients": int(len(pat)), "n_malignant": int(pat["y"].sum()),
           "auc": float(roc_auc_score(pat["y"], pat["score"])), "auc_ci": [float(lo), float(hi)], "scenarios": []}
    for slots in SLOT_SCENARIOS:
        b = slots / DAILY_ARRIVALS
        for p in PREVALENCES:
            sens, ppv, fr = at_budget(pat["y"], pat["score"], p, b)
            s_lo, s_hi = boot_ci(pat["y"], pat["score"], lambda yy, ss: at_budget(yy, ss, p, b)[0], n=500)
            out["scenarios"].append({"slots": slots, "prevalence": p, "ceiling": min(1.0, b / p),
                                     "sensitivity": float(sens), "sens_ci": [float(s_lo), float(s_hi)],
                                     "ppv": float(ppv), "flag_rate": float(fr)})
    return out


def primary(rep: dict) -> dict:
    return next(r for r in rep["scenarios"] if (r["slots"], r["prevalence"]) == PRIMARY)


def paired_auc_diff(y, s_a, s_b, n=2000, seed=42):
    """Bootstrap CI for AUC(a) - AUC(b) on the same patients."""
    rng = np.random.default_rng(seed)
    y, s_a, s_b = map(np.asarray, (y, s_a, s_b))
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    d = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        d.append(roc_auc_score(y[idx], s_a[idx]) - roc_auc_score(y[idx], s_b[idx]))
    return float(np.mean(d)), np.percentile(d, [2.5, 97.5]).tolist()
