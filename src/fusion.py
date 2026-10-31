"""Late fusion of the image model and the questionnaire by weighted average of percentile ranks.

Nothing is fitted: each patient's image score and questionnaire score become ranks within the queue, and the
patient score is w * image_rank + (1 - w) * questionnaire_rank. Ranks make the two scales comparable and keep
the image model's in-sample PAD-train scores out of the picture entirely. The project uses w = 0.6.
"""
import math

import pandas as pd


def fuse(image: pd.Series, questionnaire: pd.Series, w_image: float = 0.6) -> pd.Series:
    """Both series indexed by patient_id; returns the fused score for the patients present in both."""
    common = image.index.intersection(questionnaire.index)
    return w_image * image[common].rank(pct=True) + (1 - w_image) * questionnaire[common].rank(pct=True)


# One policy shared by both daily-list builders. These are descriptive score
# bands, NOT calibrated probabilities or clinically validated thresholds.
BAND_CUTOFFS = ((0.90, "very high"), (0.70, "high"), (0.40, "moderate"), (0.0, "low"))


def rank_band(pct_rank: float) -> str:
    """Return a descriptive band for a queue-relative rank/fused score in [0, 1]."""
    score = float(pct_rank)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"rank_band expects a finite percentile score in [0, 1], got {pct_rank!r}")
    return next(band for cutoff, band in BAND_CUTOFFS if score >= cutoff)
