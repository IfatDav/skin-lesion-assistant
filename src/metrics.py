"""Metrics on top of 6-class probabilities: per-class recall, cancer score (charter M2) and urgency score."""
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score

from src import config as C

CANCER_IDX = [C.CLASSES.index(c) for c in sorted(C.CANCER)]
URGENT_IDX = [C.CLASSES.index(c) for c, t in C.TIER.items() if t == "urgent"]


def cancer_score(probs: np.ndarray) -> np.ndarray:
    return probs[:, CANCER_IDX].sum(1)


def urgent_score(probs: np.ndarray) -> np.ndarray:
    return probs[:, URGENT_IDX].sum(1)


def threshold_for_sensitivity(score: np.ndarray, positive: np.ndarray, target: float = 0.90) -> float:
    pos = np.sort(score[positive == 1])
    return float(pos[int(np.floor((1 - target) * len(pos)))])


def summarize(probs: np.ndarray, y: np.ndarray, cancer_thr: float | None = None) -> dict:
    """y: class index per image, in C.CLASSES order."""
    pred = probs.argmax(1)
    recall = {c: float((pred[y == k] == k).mean()) for k, c in enumerate(C.CLASSES) if (y == k).any()}
    cs, us = cancer_score(probs), urgent_score(probs)
    is_cancer = np.isin(y, CANCER_IDX).astype(int)
    is_urgent = np.isin(y, URGENT_IDX).astype(int)
    out = {
        "n_images": int(len(y)),
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float(np.mean(list(recall.values()))),
        "recall_per_class": recall,
        "support_per_class": {c: int((y == k).sum()) for k, c in enumerate(C.CLASSES)},
        "cancer_auc": float(roc_auc_score(is_cancer, cs)),
        "urgent_auc": float(roc_auc_score(is_urgent, us)),
        "confusion_matrix": confusion_matrix(y, pred, labels=list(range(len(C.CLASSES)))).tolist(),
    }
    if cancer_thr is not None:
        flagged = cs >= cancer_thr
        mel = y == C.CLASSES.index("MEL")
        out.update({
            "cancer_threshold": cancer_thr,
            "cancer_sensitivity": float(flagged[is_cancer == 1].mean()),
            "cancer_specificity": float((~flagged[is_cancer == 0]).mean()),
            "flag_rate": float(flagged.mean()),
            "melanoma_caught": f"{int(flagged[mel].sum())}/{int(mel.sum())}",
        })
    return out


def patient_level(score: np.ndarray, positive: np.ndarray, patients: np.ndarray, thr: float) -> dict:
    """A patient is positive if any image is cancer; the patient score is the max over their images."""
    import pandas as pd
    g = pd.DataFrame({"p": patients, "s": score, "y": positive}).groupby("p").agg(s=("s", "max"), y=("y", "max"))
    flagged = g["s"] >= thr
    return {"n_patients": int(len(g)), "n_cancer_patients": int(g["y"].sum()),
            "cancer_auc": float(roc_auc_score(g["y"], g["s"])),
            "cancer_sensitivity": float(flagged[g["y"] == 1].mean()),
            "cancer_specificity": float((~flagged[g["y"] == 0]).mean())}
