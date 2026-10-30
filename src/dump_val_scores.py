"""Per-patient validation scores, for the M1 queue simulation.

Writes what the evaluation already computes; it fits nothing and touches no test patient.
    python -m src.dump_val_scores
"""
import json

import numpy as np
import pandas as pd

from src import config as C
from src.data import load_index, make_loader
from src.fusion import fuse
from src.metrics import cancer_score
from src.model import predict
from src.operating_point import to_patients
from src.predict import load_final_models
from src.questionnaire import patient_scores

OUT = C.REPO_ROOT / "data" / "processed" / "val_scores_fused.csv"


def main():
    w = json.loads((C.MODELS_DIR / "final_model.json").read_text())["fusion"]["w_image"]
    val = load_index("pad_val")
    p = np.mean([predict(m.cuda(), make_loader(val, train=False)).numpy()
                 for m in load_final_models()], axis=0)
    img = to_patients(val["patient_id"].to_numpy(),
                      (val["cancer"] == "1").astype(int).to_numpy(), cancer_score(p))
    quest = patient_scores("val")
    common = img.index.intersection(quest.index)

    df = pd.DataFrame({"y": img["y"].loc[common],
                       "image_score": img["score"].loc[common],
                       "questionnaire_score": quest["score"].loc[common]})
    df["fused_rank_in_set"] = fuse(df["image_score"], df["questionnaire_score"], w)
    df.index.name = "patient_id"
    df.to_csv(OUT)

    print(f"wrote {OUT.relative_to(C.REPO_ROOT)}")
    print(f"{len(df)} patients, {int(df['y'].sum())} malignant  (expected 220, 103)")
    print(df.head())


if __name__ == "__main__":
    main()
