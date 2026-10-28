"""Late fusion of the image model and the questionnaire by weighted average of percentile ranks.

Nothing is fitted: each patient's image score and questionnaire score become ranks within the queue, and the
patient score is w * image_rank + (1 - w) * questionnaire_rank. Ranks make the two scales comparable and keep
the image model's in-sample PAD-train scores out of the picture entirely. The project uses w = 0.6.
"""
import pandas as pd


def fuse(image: pd.Series, questionnaire: pd.Series, w_image: float = 0.6) -> pd.Series:
    """Both series indexed by patient_id; returns the fused score for the patients present in both."""
    common = image.index.intersection(questionnaire.index)
    return w_image * image[common].rank(pct=True) + (1 - w_image) * questionnaire[common].rank(pct=True)
