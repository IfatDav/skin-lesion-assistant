"""Cheapest image baseline: frozen DINOv2 features + logistic regression, trained on PAD training patients.

No network weights are trained. Features are extracted once (about 20 seconds on the GPU), then a logistic
regression predicts cancer vs not; patients are scored by the max over their photos.

Run:  python -m src.baseline_dinov2
"""
import json

import numpy as np
import timm
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config as C
from src.data import load_index, make_loader
from src.operating_point import primary, report, to_patients

BACKBONE = "vit_small_patch14_dinov2.lvd142m"


@torch.no_grad()
def features(model, df) -> np.ndarray:
    out = []
    for x, _ in make_loader(df, train=False):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model(x.cuda(non_blocking=True)).float().cpu())
    return torch.cat(out).numpy()


def main():
    model = timm.create_model(BACKBONE, pretrained=True, num_classes=0, img_size=C.IMG_SIZE).cuda().eval()
    tr, va = load_index("pad_train"), load_index("pad_val")
    Xtr, Xva = features(model, tr), features(model, va)
    ytr, yva = (tr["cancer"] == "1").astype(int).to_numpy(), (va["cancer"] == "1").astype(int).to_numpy()

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1, class_weight="balanced"))
    clf.fit(Xtr, ytr)
    pat = to_patients(va["patient_id"].to_numpy(), yva, clf.predict_proba(Xva)[:, 1])
    rep = report(pat)
    rep["backbone"] = BACKBONE
    rep["n_train_images"] = int(len(tr))
    (C.REPORTS_DIR / "baseline_dinov2_pad_val.json").write_text(json.dumps(rep, indent=1))
    p = primary(rep)
    print(f"DINOv2 + logistic regression | PAD val patient AUC {rep['auc']:.3f} "
          f"({rep['auc_ci'][0]:.3f}-{rep['auc_ci'][1]:.3f}) | sens @10 slots {p['sensitivity']:.2f}, PPV {p['ppv']:.2f}")


if __name__ == "__main__":
    main()
