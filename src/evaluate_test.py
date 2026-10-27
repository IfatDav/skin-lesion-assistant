"""The single run on PAD test patients. Written and committed before the run; nothing may change after it.

What is reported was fixed in reports/decisions.md before the run:
  - primary: the final model (60% image + 40% questionnaire), patient AUC with 95% CI and the share of cancer
    patients in 10 urgent slots per 100 arrivals at 10% prevalence;
  - for comparison only: image alone, questionnaire alone, paired AUC differences;
  - the same on patients whose lesions were all biopsied (PAD's unbiopsied lesions are easy benign cases).
It also writes a blind ink-labelling sheet for the test photos (no diagnosis, no score) for the planned
post-hoc shortcut check with src.ink_effect.

Run once, after the team approves the decisions:  python -m src.evaluate_test --frozen
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C
from src.data import load_index, make_loader, resolve_path
from src.fusion import fuse
from src.metrics import cancer_score
from src.model import predict
from src.operating_point import paired_auc_diff, primary, report, to_patients
from src.predict import load_final_models
from src.questionnaire import patient_scores

OUT = C.REPORTS_DIR / "test_results.json"
INK_SHEET = C.REPO_ROOT / "evaluation" / "ink_labeling_test.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", action="store_true", help="confirms the team approved reports/decisions.md")
    if not ap.parse_args().frozen:
        raise SystemExit("Refusing to load PAD test patients without --frozen (see reports/decisions.md).")
    if OUT.exists():
        raise SystemExit(f"{OUT.relative_to(C.REPO_ROOT)} already exists: the test set is used once.")

    manifest = json.loads((C.MODELS_DIR / "final_model.json").read_text())
    w = manifest["fusion"]["w_image"]
    test = load_index("pad_test")
    loader = make_loader(test, train=False)
    p = np.mean([predict(m.cuda(), loader).numpy() for m in load_final_models()], axis=0)
    assert not np.isnan(p).any()
    img = to_patients(test["patient_id"].to_numpy(), (test["cancer"] == "1").astype(int).to_numpy(), cancer_score(p))
    quest = patient_scores("test")
    common = img.index.intersection(quest.index)
    y = img["y"].loc[common]
    scores = {"image": img["score"].loc[common], "questionnaire": quest["score"].loc[common]}
    scores["final_fused"] = fuse(scores["image"], scores["questionnaire"], w)

    def block(keep):
        yy = y.loc[keep]
        s = {k: v.loc[keep] for k, v in scores.items()}
        s["final_fused"] = fuse(s["image"], s["questionnaire"], w)  # ranks within the evaluated queue
        out = {"n_patients": int(len(keep)), "n_malignant": int(yy.sum())}
        for k, v in s.items():
            rep = report(pd.DataFrame({"y": yy, "score": v}), n_boot=1000)
            out[k] = {"auc": rep["auc"], "auc_ci": rep["auc_ci"], "primary": primary(rep), "scenarios": rep["scenarios"]}
        for a, b in [("final_fused", "image"), ("final_fused", "questionnaire"), ("image", "questionnaire")]:
            m, c = paired_auc_diff(yy, s[a], s[b])
            out[f"{a}_minus_{b}"] = {"mean": m, "ci": c}
        return out

    meta = pd.read_csv(C.META_DIR / "PAD_UFES_20_metadata.csv")
    all_biopsied = meta["biopsed"].astype(str).str.upper().eq("TRUE").groupby(meta["patient_id"]).all()
    res = {"manifest": manifest, "all_patients": block(common),
           "biopsied_only": block(common[all_biopsied.loc[common].to_numpy()])}
    OUT.write_text(json.dumps(res, indent=1, default=float))

    sheet = test[["image_id", "split"]].copy()
    sheet["photo"] = [Path(os.path.relpath(resolve_path(r), C.REPO_ROOT)).as_posix() for _, r in test.iterrows()]
    sheet["has_ink"], sheet["notes"] = "", ""
    sheet.sample(frac=1, random_state=C.SEED).to_csv(INK_SHEET, index=False)

    for name, r in [("all patients", res["all_patients"]), ("biopsied only", res["biopsied_only"])]:
        print(f"{name}: {r['n_patients']} patients, {r['n_malignant']} with cancer")
        for k in scores:
            print(f"  {k:<14} AUC {r[k]['auc']:.3f} ({r[k]['auc_ci'][0]:.3f}-{r[k]['auc_ci'][1]:.3f}) | "
                  f"sens @10 slots {r[k]['primary']['sensitivity']:.2f} | PPV {r[k]['primary']['ppv']:.2f}")
    print(f"wrote {OUT.relative_to(C.REPO_ROOT)} and {INK_SHEET.relative_to(C.REPO_ROOT)}")


if __name__ == "__main__":
    main()
