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


def summarize(probs: np.ndarray, y: np.ndarray) -> dict:
    """y: class index per image, in C.CLASSES order."""
    pred = probs.argmax(1)
    recall = {c: float((pred[y == k] == k).mean()) for k, c in enumerate(C.CLASSES) if (y == k).any()}
    cs, us = cancer_score(probs), urgent_score(probs)
    is_cancer = np.isin(y, CANCER_IDX).astype(int)
    is_urgent = np.isin(y, URGENT_IDX).astype(int)
    return {
        "n_images": int(len(y)),
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float(np.mean(list(recall.values()))),
        "recall_per_class": recall,
        "support_per_class": {c: int((y == k).sum()) for k, c in enumerate(C.CLASSES)},
        "cancer_auc": float(roc_auc_score(is_cancer, cs)),
        "urgent_auc": float(roc_auc_score(is_urgent, us)),
        "confusion_matrix": confusion_matrix(y, pred, labels=list(range(len(C.CLASSES)))).tolist(),
    }
