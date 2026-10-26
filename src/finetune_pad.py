"""Step 4 of the plan: fine-tune the ISIC-trained model on PAD training patients (phone photos).

Early stages stay frozen and the learning rate is ten times lower than on ISIC, because 1,430 images are few
enough for the network to memorise. The best epoch is chosen on PAD validation patients (patient-level AUC).
PAD test patients are never loaded.

--scc-boost multiplies the SCC class weight: after plain fine-tuning, SCC photos were often called actinic
keratosis (non-cancer), because PAD has four times more AK than SCC.

Run:  python -m src.finetune_pad [--seed 42] [--scc-boost 2]
      -> models/convnext_pad_s42.pt (or convnext_pad_s42_scc2.pt)
"""
import argparse
import json
import math
import random
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from src import config as C
from src.data import load_index, make_loader
from src.metrics import cancer_score
from src.model import create_model, make_optimizer, predict, warmup_cosine
from src.operating_point import to_patients
from src.train import class_weights, run_epoch


def checkpoint_name(seed: int, scc_boost: float) -> str:
    return f"convnext_pad_s{seed}" + (f"_scc{scc_boost:g}" if scc_boost != 1 else "") + ".pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--scc-boost", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-5)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    train_df, val_df = load_index("pad_train"), load_index("pad_val")
    assert not set(train_df["patient_id"]) & set(val_df["patient_id"])
    train_dl, val_dl = make_loader(train_df, train=True), make_loader(val_df, train=False)
    y_train = np.array([C.CLASSES.index(l) for l in train_df["label6"]])
    y_val_cancer = (val_df["cancer"] == "1").astype(int).to_numpy()

    start = C.MODELS_DIR / f"convnext_milk_s{args.seed}.pt"
    model = create_model(pretrained=False)
    model.load_state_dict(torch.load(start, map_location="cpu")["state_dict"])
    model = model.cuda()

    w = class_weights(y_train)
    w[C.CLASSES.index("SCC")] *= args.scc_boost
    w = w / w.mean()
    crit = torch.nn.CrossEntropyLoss(weight=w.cuda(), label_smoothing=0.1)
    opt = make_optimizer(model, "late", lr=args.lr)
    sched = warmup_cosine(opt, args.epochs * len(train_dl), max(1, len(train_dl) // 2))

    def val_auc():
        pat = to_patients(val_df["patient_id"].to_numpy(), y_val_cancer, cancer_score(predict(model, val_dl).numpy()))
        return roc_auc_score(pat["y"], pat["score"])

    log = {"config": vars(args) | {"start_checkpoint": start.name, "trainable": "stages 2-3 + head",
                                   "n_train_images": len(train_df), "n_train_patients": int(train_df["patient_id"].nunique()),
                                   "n_val_patients": int(val_df["patient_id"].nunique()),
                                   "class_weights": dict(zip(C.CLASSES, map(float, w.round(decimals=3))))},
           "epochs": []}
    log["val_patient_auc_before"] = round(val_auc(), 4)
    print(json.dumps(log["config"]), f"\nbefore fine-tuning: PAD val patient AUC {log['val_patient_auc_before']}", flush=True)

    best, best_ep, stale = -math.inf, 0, 0
    out = C.MODELS_DIR / checkpoint_name(args.seed, args.scc_boost)
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        loss = run_epoch(model, train_dl, opt, sched, crit)
        auc = val_auc()
        row = {"epoch": ep, "train_loss": round(loss, 4), "val_patient_auc": round(auc, 4), "seconds": round(time.time() - t0)}
        log["epochs"].append(row)
        print(row, flush=True)
        if auc > best:
            best, best_ep, stale = auc, ep, 0
            torch.save({"state_dict": model.state_dict(), "classes": C.CLASSES, "epoch": ep}, out)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stop: no improvement for {args.patience} epochs", flush=True)
                break
    log["best_epoch"] = best_ep
    (C.REPORTS_DIR / "logs").mkdir(parents=True, exist_ok=True)
    (C.REPORTS_DIR / "logs" / out.with_suffix(".json").name.replace("convnext_", "finetune_")).write_text(json.dumps(log, indent=1))
    print(f"best epoch {best_ep} (val patient AUC {best:.4f}), checkpoint {out.relative_to(C.REPO_ROOT)}", flush=True)


if __name__ == "__main__":
    main()
