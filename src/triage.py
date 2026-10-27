"""Ranks a day's queue with the final model: 60% image rank + 40% questionnaire rank (team decision, see
reports/decisions.md).

    from src.triage import Triage
    t = Triage()
    t.rank([{"patient_id": "a", "photos": ["a1.jpg"], "age": 61, "region": "FACE",
             "itch": True, "grew": True, "hurt": False, "changed": False, "bleed": True, "elevation": True},
            ...])

Ranks are taken within the queue, so the queue needs several patients; one patient alone has no rank.
A missing answer is filled the same way as in training (most frequent value); age and region are required.
"""
import json

import pandas as pd

from src import questionnaire
from src.fusion import fuse
from src.predict import MANIFEST, VisualRiskModel


class Triage:
    def __init__(self):
        cfg = json.loads(MANIFEST.read_text())
        self.w_image = cfg["fusion"]["w_image"]
        self.slots_per_100 = cfg["operating_point"]["slots_per_100"]
        self.image = VisualRiskModel()
        self.quest = questionnaire.fit_model()

    def rank(self, queue: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(queue).set_index("patient_id")
        for c in questionnaire.SYMPTOMS:
            if c not in df:
                df[c] = None
        img = {pid: self.image.predict(row["photos"]) for pid, row in df.iterrows()}
        df["image_score"] = pd.Series({pid: r["cancer_score"] for pid, r in img.items()})
        df["questionnaire_score"] = questionnaire.score(self.quest, df.reset_index())
        df["score"] = fuse(df["image_score"], df["questionnaire_score"], self.w_image)
        df["most_suspicious_photo"] = pd.Series({pid: r["most_suspicious_photo"] for pid, r in img.items()})
        df["most_likely_class"] = pd.Series({pid: r["most_likely_class"] for pid, r in img.items()})
        out = df.sort_values("score", ascending=False)
        n_urgent = max(1, round(len(out) * self.slots_per_100 / 100))
        out["urgent"] = [i < n_urgent for i in range(len(out))]
        return out[["score", "urgent", "image_score", "questionnaire_score", "most_likely_class", "most_suspicious_photo"]]


if __name__ == "__main__":
    import sys
    print(Triage().rank(json.loads(open(sys.argv[1], encoding="utf-8").read())).to_string())
