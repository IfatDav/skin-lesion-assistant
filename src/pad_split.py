"""The team's frozen PAD-UFES-20 patient split (notebook 02, Cell 4), reproduced and verified.

The split is defined by the notebook's code: patients stratified by melanoma / other malignant / non-malignant,
20% test, then 20% of the rest as validation, random_state=42. If the original manifest is placed at
data/splits/pad_split.csv it is used instead, after checking it against the published counts.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config as C

TEAM_MANIFEST = C.SPLITS_DIR / "pad_split.csv"

# Printed by notebook 02 (Cell 4); any drift means the split is not the team's.
EXPECTED = {
    "test": {"images": 501, "patients": 275, "malignant_patients": 129, "melanoma_patients": 7},
    "train": {"images": 1440, "patients": 878, "malignant_patients": 411, "melanoma_patients": 23},
    "val": {"images": 357, "patients": 220, "malignant_patients": 103, "melanoma_patients": 6},
}


def _patients(meta: pd.DataFrame) -> pd.DataFrame:
    is_mal = meta["diagnostic"].isin(["MEL", "BCC", "SCC"]).astype(int)
    is_mel = (meta["diagnostic"] == "MEL").astype(int)
    p = (meta.assign(is_mal=is_mal, is_mel=is_mel).groupby("patient_id")
             .agg(has_malignant=("is_mal", "max"), has_melanoma=("is_mel", "max")).reset_index())
    p["stratum"] = np.select([p["has_melanoma"] == 1, p["has_malignant"] == 1],
                             ["melanoma", "other_malignant"], default="non_malignant")
    return p


def patient_split(meta: pd.DataFrame) -> dict:
    """patient_id -> 'train' | 'val' | 'test', verified against notebook 02's counts."""
    if TEAM_MANIFEST.exists():
        frozen = pd.read_csv(TEAM_MANIFEST, dtype=str)
        split = frozen.groupby("patient_id")["split"].first().to_dict()
    else:
        p = _patients(meta)
        dev, test = train_test_split(p, test_size=0.20, random_state=C.SEED, stratify=p["stratum"])
        train, val = train_test_split(dev, test_size=0.20, random_state=C.SEED, stratify=dev["stratum"])
        split = {**{x: "train" for x in train["patient_id"]}, **{x: "val" for x in val["patient_id"]},
                 **{x: "test" for x in test["patient_id"]}}

    p = _patients(meta).assign(split=lambda d: d["patient_id"].map(split))
    imgs = meta["patient_id"].map(split).value_counts()
    for s, exp in EXPECTED.items():
        got = {"images": int(imgs.get(s, 0)), "patients": int((p["split"] == s).sum()),
               "malignant_patients": int(p.loc[p["split"] == s, "has_malignant"].sum()),
               "melanoma_patients": int(p.loc[p["split"] == s, "has_melanoma"].sum())}
        assert got == exp, f"PAD split '{s}' does not match notebook 02: {got} != {exp}"
    return split
