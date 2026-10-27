"""Analyse the manually labelled ink sample (evaluation/ink_labeling_sample.csv, see evaluation/INK_LABELING.md).

A shortcut shows up as inked photos getting higher cancer scores than un-inked photos of the same diagnosis.

Run:  python -m src.ink_effect                                        # development sample
      python -m src.ink_effect evaluation/ink_labeling_test.csv      # test photos, after the frozen test run
"""
import json
import sys

import pandas as pd
from scipy.stats import mannwhitneyu

from src import config as C
from src.data import load_index, resolve_path
from src.predict import VisualRiskModel

SAMPLE = C.REPO_ROOT / "evaluation" / "ink_labeling_sample.csv"


def main(sample=SAMPLE):
    sample = C.REPO_ROOT / sample
    lab = pd.read_csv(sample, dtype=str).fillna("")
    lab = lab[lab["has_ink"].isin(["0", "1"])]
    if lab.empty:
        raise SystemExit(f"No labels yet: fill the has_ink column in {sample.relative_to(C.REPO_ROOT)}")
    idx = load_index(sorted(set(lab["split"]))).set_index("image_id")
    lab = lab.join(idx[["label6", "cancer", "source", "image_path"]], on="image_id")

    vrm = VisualRiskModel()
    lab["cancer_score"] = [vrm.predict([resolve_path(r)])["cancer_score"] for _, r in lab.iterrows()]

    ink = lab["has_ink"] == "1"
    out = {"labelled": int(len(lab)), "with_ink": int(ink.sum()),
           "ink_rate_cancer": float(ink[lab["cancer"] == "1"].mean()),
           "ink_rate_non_cancer": float(ink[lab["cancer"] == "0"].mean()), "within_class": {}}
    for c, g in lab.groupby("label6"):
        a, b = g.loc[g["has_ink"] == "1", "cancer_score"], g.loc[g["has_ink"] == "0", "cancer_score"]
        if len(a) >= 3 and len(b) >= 3:
            out["within_class"][c] = {"n_ink": int(len(a)), "n_no_ink": int(len(b)),
                                      "median_score_ink": float(a.median()), "median_score_no_ink": float(b.median()),
                                      "mannwhitney_p": float(mannwhitneyu(a, b).pvalue)}
    (C.REPORTS_DIR / f"ink_effect{'_test' if 'pad_test' in set(lab['split']) else ''}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
