"""Train the image model on the ISIC clinical pool (steps 1-2 of the plan). PAD is not touched here.

Run:  python -m src.train [--seed 42]      -> models/convnext_milk_s42.pt
"""
import argparse
import json
import math
import random
import time

import numpy as np
import torch

from src import config as C
from src.data import load_index, make_loader
from src.metrics import summarize
from src.model import create_model, make_optimizer, predict, warmup_cosine


def class_weights(labels) -> torch.Tensor:
    counts = np.bincount(labels, minlength=len(C.CLASSES))
    w = 1 / np.sqrt(counts)
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def run_epoch(model, loader, opt, sched, crit) -> float:
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = crit(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()
        total += loss.item() * len(y)
        n += len(y)
    return total / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head-epochs", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--layer-decay", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=C.SEED)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    train_df, val_df = load_index("milk_train"), load_index("milk_val")
    train_dl, val_dl = make_loader(train_df, train=True), make_loader(val_df, train=False)
    y_train = np.array([C.CLASSES.index(l) for l in train_df["label6"]])
    y_val = np.array([C.CLASSES.index(l) for l in val_df["label6"]])

    w = class_weights(y_train)
    crit = torch.nn.CrossEntropyLoss(weight=w.cuda(), label_smoothing=0.1)
    model = create_model().cuda()
    log = {"config": vars(args) | {"backbone": "convnext_tiny.fb_in22k_ft_in1k", "img_size": C.IMG_SIZE,
                                   "n_train": len(train_df), "n_val": len(val_df),
                                   "class_weights": dict(zip(C.CLASSES, map(float, w.round(decimals=3))))},
           "epochs": []}
    print(json.dumps(log["config"]), flush=True)

    def step(phase, ep, loss, t0):
        m = summarize(predict(model, val_dl).numpy(), y_val)
        row = {"phase": phase, "epoch": ep, "train_loss": round(loss, 4), "seconds": round(time.time() - t0),
               "val_cancer_auc": round(m["cancer_auc"], 4), "val_balanced_acc": round(m["balanced_accuracy"], 4),
               "val_accuracy": round(m["accuracy"], 4)}
        log["epochs"].append(row)
        print(row, flush=True)
        return m

    # Step 1: train only the new head so its random start does not disturb the pretrained weights.
    opt = make_optimizer(model, "head", lr=1e-3)
    for ep in range(1, args.head_epochs + 1):
        t0 = time.time()
        step("head", ep, run_epoch(model, train_dl, opt, None, crit), t0)

    # Step 2: unfreeze everything, earlier layers at smaller learning rates; keep the best epoch.
    opt = make_optimizer(model, "all", lr=args.lr, layer_decay=args.layer_decay)
    sched = warmup_cosine(opt, args.epochs * len(train_dl), len(train_dl))
    best, best_ep, stale = -math.inf, 0, 0
    C.MODELS_DIR.mkdir(exist_ok=True)
    ckpt = C.MODELS_DIR / f"convnext_milk_s{args.seed}.pt"
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        m = step("full", ep, run_epoch(model, train_dl, opt, sched, crit), t0)
        score = m["cancer_auc"] + m["balanced_accuracy"]
        if score > best:
            best, best_ep, stale = score, ep, 0
            torch.save({"state_dict": model.state_dict(), "classes": C.CLASSES, "epoch": ep}, ckpt)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stop: no improvement for {args.patience} epochs", flush=True)
                break
    log["best_full_epoch"] = best_ep
    C.REPORTS_DIR.mkdir(exist_ok=True)
    (C.REPORTS_DIR / "logs").mkdir(exist_ok=True)
    (C.REPORTS_DIR / "logs" / f"train_milk_s{args.seed}.json").write_text(json.dumps(log, indent=1))
    print(f"best epoch {best_ep}, checkpoint {ckpt.relative_to(C.REPO_ROOT)}", flush=True)


if __name__ == "__main__":
    main()
