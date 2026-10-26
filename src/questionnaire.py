"""Questionnaire-only baseline (notebook 02, Cell 8b, variant D), reproduced so the image model can be compared
on the same validation patients. Uses only the app's mandatory fields; missing answers are imputed inside the
pipeline and no missingness indicator is given to the model.
"""
import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config as C
from src.operating_point import to_patients
from src.pad_split import patient_split

SYMPTOMS = ["itch", "grew", "hurt", "changed", "bleed", "elevation"]
NUM, CAT = ["age"], ["region"] + SYMPTOMS


def _prep(df):
    X = df[NUM + CAT].copy()
    for c in CAT:
        X[c] = X[c].map(lambda v: np.nan if pd.isna(v) else str(v).strip().upper()).astype(object)
    return X


def val_patient_scores() -> pd.DataFrame:
    pad = pd.read_csv(C.META_DIR / "PAD_UFES_20_metadata.csv").replace({"UNK": np.nan, "": np.nan})
    pad["split"] = pad["patient_id"].map(patient_split(pad.astype({"diagnostic": str})))
    pad["is_malignant"] = pad["diagnostic"].isin(["MEL", "BCC", "SCC"]).astype(int)
    pad["lesion_key"] = pad["patient_id"].astype(str) + "::" + pad["lesion_id"].astype(str)
    lesions = pad[pad["split"].isin(["train", "val"])].sort_values("img_id").drop_duplicates("lesion_key")
    tr, va = lesions[lesions["split"] == "train"], lesions[lesions["split"] == "val"]
    model = make_pipeline(
        make_column_transformer(
            (make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUM),
            (make_pipeline(SimpleImputer(strategy="most_frequent"),
                           OneHotEncoder(handle_unknown="ignore", min_frequency=10)), CAT)),
        LogisticRegression(max_iter=2000, class_weight="balanced"))
    model.fit(_prep(tr), tr["is_malignant"])
    return to_patients(va["patient_id"].to_numpy(), va["is_malignant"].to_numpy(),
                       model.predict_proba(_prep(va))[:, 1])
