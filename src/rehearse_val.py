"""Dress rehearsal for the single test run: the exact evaluate_test path, on PAD *validation* patients.

    python -m src.rehearse_val

It touches no test data and writes no files. It walks the same code path as src.evaluate_test -- the same
index, loader, checkpoint, cancer score, patient aggregation, questionnaire model and fusion -- and checks
the result against the numbers published in reports/final_results.json.

If it passes, this environment reproduces the published validation numbers and the single test run can go
ahead. If it fails, something here differs from the machine that produced those numbers (GPU precision,
scikit-learn version, a different image set) and the test run should wait: the test set is used once.
"""
import json

import numpy as np
import pandas as pd
import torch

from src import config as C
from src.data import load_index, make_loader
from src.fusion import fuse
from src.metrics import cancer_score
from src.model import predict
from src.operating_point import paired_auc_diff, primary, report, to_patients
from src.predict import load_final_models
from src.questionnaire import patient_scores

# reports/final_results.json, the run of 26.10.2026 on 220 PAD validation patients
PUBLISHED_AUC = {"image": 0.9556, "questionnaire": 0.8668, "final_fused": 0.9542}
PUBLISHED_SENS = {"image": 0.602, "questionnaire": 0.524, "final_fused": 0.612}
PUBLISHED_PAIRED = 0.0888  # image - questionnaire
AUC_TOL, SENS_TOL = 0.005, 0.02

EXPECT_IMAGES, EXPECT_PATIENTS, EXPECT_MALIGNANT = 357, 220, 103


def main() -> int:
    has_cuda = torch.cuda.is_available()
    print(f"torch {torch.__version__} | cuda {has_cuda} | "
          f"{torch.cuda.get_device_name(0) if has_cuda else 'NO GPU'} | "
          f"bf16 {torch.cuda.is_bf16_supported() if has_cuda else False}")
    if not has_cuda:
        print("\nNo CUDA device. src.model.predict and src.evaluate_test both require one.")
        return 1

    import sklearn
    print(f"scikit-learn {sklearn.__version__} (models/questionnaire.joblib was written with 1.9.1)")
    print(f"frozen PAD manifest at data/splits/pad_split.csv: {(C.SPLITS_DIR / 'pad_split.csv').exists()}")

    manifest = json.loads((C.MODELS_DIR / "final_model.json").read_text())
    w = manifest["fusion"]["w_image"]

    val = load_index("pad_val")
    print(f"\nvalidation images: {len(val)} (expected {EXPECT_IMAGES})")
    assert len(val) == EXPECT_IMAGES, "wrong image count: this is not the frozen validation split"

    p = np.mean([predict(m.cuda(), make_loader(val, train=False)).numpy()
                 for m in load_final_models()], axis=0)
    assert not np.isnan(p).any(), "NaN predictions -- see the pin_memory note in src/data.py"

    img = to_patients(val["patient_id"].to_numpy(),
                      (val["cancer"] == "1").astype(int).to_numpy(), cancer_score(p))
    quest = patient_scores("val")
    common = img.index.intersection(quest.index)
    y = img["y"].loc[common]
    scores = {"image": img["score"].loc[common], "questionnaire": quest["score"].loc[common]}
    scores["final_fused"] = fuse(scores["image"], scores["questionnaire"], w)

    print(f"patients: {len(common)} (expected {EXPECT_PATIENTS}) | "
          f"malignant: {int(y.sum())} (expected {EXPECT_MALIGNANT})\n")
    ok = len(common) == EXPECT_PATIENTS and int(y.sum()) == EXPECT_MALIGNANT

    for k, v in scores.items():
        rep = report(pd.DataFrame({"y": y, "score": v}), n_boot=1000)
        prim = primary(rep)
        d_auc = rep["auc"] - PUBLISHED_AUC[k]
        d_sens = prim["sensitivity"] - PUBLISHED_SENS[k]
        good = abs(d_auc) <= AUC_TOL and abs(d_sens) <= SENS_TOL
        ok &= good
        print(f"  {'OK  ' if good else 'DIFF'} {k:<14} "
              f"AUC {rep['auc']:.4f} vs {PUBLISHED_AUC[k]:.4f} ({d_auc:+.4f}) | "
              f"sens@10 slots {prim['sensitivity']:.3f} vs {PUBLISHED_SENS[k]:.3f} ({d_sens:+.3f})")

    m, ci = paired_auc_diff(y, scores["image"], scores["questionnaire"])
    print(f"       image - questionnaire  {m:+.4f} vs {PUBLISHED_PAIRED:+.4f} "
          f"| CI [{ci[0]:+.3f}, {ci[1]:+.3f}]")

    print("\nREHEARSAL PASSED. This environment reproduces the published validation numbers;\n"
          "`python -m src.evaluate_test --frozen` can run." if ok else
          "\nREHEARSAL FAILED. Do not run the test yet -- resolve the differences above first.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
