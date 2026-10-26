"""Final evaluation on PAD validation patients. PAD test patients are never loaded.

Applies the decision rules written down before the runs:
  1. SCC weight x2 is adopted only if, averaged over the 3 seeds, fewer SCC photos are called AK and the
     patient AUC drops by no more than 0.01.
  2. The final model is the seed-42 run of the adopted variant, not the best seed.
  3. Averaging the 3 seeds is adopted only if its paired AUC gain over the single model excludes zero.
  4. Fusion with the questionnaire uses fixed weights, 60% image / 40% questionnaire.

Writes reports/final_results.json, reports/figures/*.png and models/final_model.json.

Run:  python -m src.evaluate
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from src import config as C
from src.data import load_index, make_loader
from src.fusion import fuse
from src.metrics import cancer_score, summarize
from src.model import create_model, predict
from src.operating_point import (DAILY_ARRIVALS, PRIMARY, at_budget, flagged_mask, paired_auc_diff, primary,
                                 report, to_patients)
from src.questionnaire import val_patient_scores

SEEDS = [42, 1, 2]
VARIANTS = {"plain": "", "scc_x2": "_scc2"}
FUSION_W_IMAGE = 0.6
FIG = C.REPORTS_DIR / "figures"


def probs_for(ckpt_names, loader) -> np.ndarray:
    out = []
    for name in ckpt_names:
        m = create_model(pretrained=False)
        m.load_state_dict(torch.load(C.MODELS_DIR / name, map_location="cpu")["state_dict"])
        out.append(predict(m.cuda(), loader).numpy())
        del m
        torch.cuda.empty_cache()
    return np.mean(out, axis=0)


def curve(pat: pd.DataFrame, prevalence=0.10, max_slots=30) -> list:
    return [round(float(at_budget(pat["y"], pat["score"], prevalence, s / DAILY_ARRIVALS)[0]), 4)
            for s in range(1, max_slots + 1)]


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    val = load_index("pad_val")
    assert set(val["split"]) == {"pad_val"}
    loader = make_loader(val, train=False)
    pid = val["patient_id"].to_numpy()
    y_cancer = (val["cancer"] == "1").astype(int).to_numpy()
    y6 = np.array([C.CLASSES.index(l) for l in val["label6"]])
    SCC, AK = C.CLASSES.index("SCC"), C.CLASSES.index("AK")
    mel_pat = to_patients(pid, (val["label6"] == "MEL").astype(int).to_numpy(), np.zeros(len(val)))["y"]
    slots, prev = PRIMARY

    def evaluate_probs(p):
        pat = to_patients(pid, y_cancer, cancer_score(p))
        rep = report(pat, n_boot=1000)
        pred = p.argmax(1)
        flags = flagged_mask(pat["y"], pat["score"], prev, slots / DAILY_ARRIVALS)
        rep.update({"scc_recall": float((pred[y6 == SCC] == SCC).mean()),
                    "scc_called_ak": int(((y6 == SCC) & (pred == AK)).sum()), "n_scc": int((y6 == SCC).sum()),
                    "melanoma_patients_flagged": int(flags[mel_pat.loc[pat.index].to_numpy() == 1].sum()),
                    "n_melanoma_patients": int(mel_pat.sum())})
        return rep, pat

    res = {"pad_val_patients": int(len(np.unique(pid))), "rules": __doc__.split("Writes")[0].strip()}

    # --- seeds x variants ---------------------------------------------------------------------------------
    runs, probs = {}, {}
    for v, suffix in VARIANTS.items():
        for s in SEEDS:
            name = f"convnext_pad_s{s}{suffix}.pt"
            probs[(v, s)] = probs_for([name], loader)
            rep, _ = evaluate_probs(probs[(v, s)])
            runs[f"{v}_s{s}"] = {k: rep[k] for k in ("auc", "auc_ci", "scc_recall", "scc_called_ak", "n_scc",
                                                    "melanoma_patients_flagged", "n_melanoma_patients")}
            runs[f"{v}_s{s}"]["sens_primary"] = primary(rep)["sensitivity"]
    res["runs"] = runs

    def agg(v):
        r = [runs[f"{v}_s{s}"] for s in SEEDS]
        return {k: {"mean": float(np.mean([x[k] for x in r])), "min": float(np.min([x[k] for x in r])),
                    "max": float(np.max([x[k] for x in r]))}
                for k in ("auc", "sens_primary", "scc_recall", "scc_called_ak", "melanoma_patients_flagged")}
    res["seed_summary"] = {v: agg(v) for v in VARIANTS}

    plain, boost = res["seed_summary"]["plain"], res["seed_summary"]["scc_x2"]
    adopt_boost = (boost["scc_called_ak"]["mean"] < plain["scc_called_ak"]["mean"]
                   and boost["auc"]["mean"] >= plain["auc"]["mean"] - 0.01)
    variant = "scc_x2" if adopt_boost else "plain"
    res["decision_scc_boost"] = {"adopted": bool(adopt_boost), "variant": variant}

    # --- single model vs average of seeds -----------------------------------------------------------------
    single_rep, single_pat = evaluate_probs(probs[(variant, 42)])
    ens_p = np.mean([probs[(variant, s)] for s in SEEDS], axis=0)
    ens_rep, ens_pat = evaluate_probs(ens_p)
    d, ci = paired_auc_diff(single_pat["y"], ens_pat["score"], single_pat["score"])
    adopt_ens = ci[0] > 0
    res["decision_ensemble"] = {"adopted": bool(adopt_ens), "auc_gain_mean": d, "auc_gain_ci": ci,
                                "ensemble_auc": ens_rep["auc"], "single_auc": single_rep["auc"]}
    final_ckpts = [f"convnext_pad_s{s}{VARIANTS[variant]}.pt" for s in (SEEDS if adopt_ens else [42])]
    (C.MODELS_DIR / "final_model.json").write_text(json.dumps({"checkpoints": final_ckpts}, indent=1))
    final_p = ens_p if adopt_ens else probs[(variant, 42)]
    final_rep, final_pat = (ens_rep, ens_pat) if adopt_ens else (single_rep, single_pat)
    final_rep["image_level_6class"] = summarize(final_p, y6)
    res["final_image_model"] = {"checkpoints": final_ckpts, **final_rep}

    # --- baselines and fusion on the same patients --------------------------------------------------------
    quest = val_patient_scores()
    res["questionnaire"] = report(quest, n_boot=1000)
    isic_p = probs_for(["convnext_milk_s42.pt"], loader)
    isic_rep, isic_pat = evaluate_probs(isic_p)
    res["image_isic_only"] = isic_rep
    dino = json.loads((C.REPORTS_DIR / "baseline_dinov2_pad_val.json").read_text())
    res["dinov2_logreg"] = dino

    fused = fuse(final_pat["score"], quest["score"], FUSION_W_IMAGE)
    fused_pat = pd.DataFrame({"y": final_pat["y"].loc[fused.index], "score": fused})
    res["fused"] = report(fused_pat, n_boot=1000)
    res["fused"]["weights"] = {"image": FUSION_W_IMAGE, "questionnaire": 1 - FUSION_W_IMAGE}
    common = fused.index
    for name, s in [("image_minus_questionnaire", final_pat["score"]), ("fused_minus_questionnaire", fused)]:
        m, c = paired_auc_diff(fused_pat["y"], s.loc[common], quest["score"].loc[common])
        res[name] = {"mean": m, "ci": c}
    m, c = paired_auc_diff(fused_pat["y"], fused, final_pat["score"].loc[common])
    res["fused_minus_image"] = {"mean": m, "ci": c}
    res["fusion_weight_sweep"] = {}
    for w in [0.0, 0.4, 0.5, 0.6, 0.7, 1.0]:
        f = fuse(final_pat["score"], quest["score"], w)
        rep = report(pd.DataFrame({"y": final_pat["y"].loc[f.index], "score": f}), n_boot=300)
        res["fusion_weight_sweep"][str(w)] = {"auc": rep["auc"], "sens_primary": primary(rep)["sensitivity"]}

    # --- where the gain comes from ------------------------------------------------------------------------
    ak_pat = set(val.loc[val["label6"] == "AK", "patient_id"])
    keep = lambda pat: ~pat.index.isin(ak_pat)
    cnt = val.groupby("patient_id").agg(n=("image_id", "size"))["n"].loc[final_pat.index]
    mean_pat = pd.DataFrame({"p": pid, "y": y_cancer, "s": cancer_score(final_p)}).groupby("p").agg(y=("y", "max"), s=("s", "mean"))
    res["diagnostics"] = {
        "isic_only_auc_without_ak_patients": float(roc_auc_score(isic_pat["y"][keep(isic_pat)], isic_pat["score"][keep(isic_pat)])),
        "final_auc_without_ak_patients": float(roc_auc_score(final_pat["y"][keep(final_pat)], final_pat["score"][keep(final_pat)])),
        "n_patients_without_ak": int(keep(final_pat).sum()),
        "final_auc_mean_aggregation": float(roc_auc_score(mean_pat["y"], mean_pat["s"])),
        "final_auc_image_level": float(roc_auc_score(y_cancer, cancer_score(final_p))),
        "images_per_patient_auc": float(roc_auc_score(final_pat["y"], cnt)),
        "mean_images_cancer": float(cnt[final_pat["y"] == 1].mean()),
        "mean_images_non_cancer": float(cnt[final_pat["y"] == 0].mean()),
    }

    # --- MSK false alarms, image model at its primary-point cutoff ----------------------------------------
    msk = load_index("msk_eval")
    benign, dark = (msk["cancer"] == "0").to_numpy(), msk["fitzpatrick"].isin(["V", "VI"]).to_numpy()
    flags = flagged_mask(final_pat["y"], final_pat["score"], prev, slots / DAILY_ARRIVALS)
    cutoff = final_pat["score"][flags].min()
    ms = cancer_score(probs_for(final_ckpts, make_loader(msk, train=False)))
    res["msk_false_alarms"] = {"cutoff": float(cutoff), "n_benign": int(benign.sum()),
                               "all_benign": float((ms[benign] >= cutoff).mean()),
                               "fitz_V_VI": float((ms[benign & dark] >= cutoff).mean()),
                               "fitz_I_IV": float((ms[benign & ~dark] >= cutoff).mean())}

    # --- curves for figures -------------------------------------------------------------------------------
    res["curves_sens_vs_slots_10pct"] = {"slots": list(range(1, 31)),
                                         "questionnaire": curve(quest), "image": curve(final_pat),
                                         "fused": curve(fused_pat), "image_isic_only": curve(isic_pat)}
    logs = C.REPORTS_DIR / "logs"
    res["training_curves_s42"] = {
        "isic": json.loads((logs / "train_milk_s42.json").read_text())["epochs"],
        "pad": json.loads((logs / f"finetune_pad_s42{VARIANTS[variant]}.json").read_text())}

    (C.REPORTS_DIR / "final_results.json").write_text(json.dumps(res, indent=1, default=float))
    make_figures(res)
    print_summary(res)


def make_figures(res):
    c = res["curves_sens_vs_slots_10pct"]
    plt.figure(figsize=(6.5, 4))
    for k, lab in [("questionnaire", "Questionnaire"), ("image_isic_only", "Image, ISIC only"),
                   ("image", "Image, fine-tuned on PAD"), ("fused", "Image 60% + questionnaire 40%")]:
        plt.plot(c["slots"], c[k], label=lab, lw=2)
    plt.plot(c["slots"], [min(1, s / 10) for s in c["slots"]], "k:", lw=1, label="Best possible")
    plt.axvline(10, color="grey", lw=0.8)
    plt.xlabel("Urgent slots per 100 arrivals"); plt.ylabel("Share of cancer patients on the list")
    plt.title("PAD validation patients, 10% prevalence"); plt.legend(fontsize=8); plt.ylim(0, 1.02)
    plt.tight_layout(); plt.savefig(FIG / "sensitivity_vs_slots.png", dpi=130); plt.close()

    cm = np.array(res["final_image_model"]["image_level_6class"]["confusion_matrix"])
    plt.figure(figsize=(5, 4.3))
    plt.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i in range(6):
        for j in range(6):
            plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=9)
    plt.xticks(range(6), C.CLASSES); plt.yticks(range(6), C.CLASSES)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Final image model, PAD validation photos")
    plt.tight_layout(); plt.savefig(FIG / "confusion_matrix_pad_val.png", dpi=130); plt.close()

    t = res["training_curves_s42"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
    isic = t["isic"]
    ax[0].plot(range(1, len(isic) + 1), [e["val_cancer_auc"] for e in isic], "o-", label="val cancer AUC")
    ax[0].plot(range(1, len(isic) + 1), [e["train_loss"] for e in isic], "s--", label="train loss")
    ax[0].axvline(2.5, color="grey", lw=0.8); ax[0].set_title("ISIC (head warm-up | full)"); ax[0].set_xlabel("epoch"); ax[0].legend(fontsize=8)
    pad = t["pad"]
    ax[1].plot([0] + [e["epoch"] for e in pad["epochs"]], [pad["val_patient_auc_before"]] + [e["val_patient_auc"] for e in pad["epochs"]], "o-", label="val patient AUC")
    ax[1].plot([e["epoch"] for e in pad["epochs"]], [e["train_loss"] for e in pad["epochs"]], "s--", label="train loss")
    ax[1].set_title("PAD fine-tuning"); ax[1].set_xlabel("epoch"); ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(FIG / "training_curves.png", dpi=130); plt.close()


def print_summary(res):
    print(f"PAD validation: {res['pad_val_patients']} patients\n")
    print("seed summary (mean [min-max]):")
    for v, s in res["seed_summary"].items():
        print(f"  {v:<7} AUC {s['auc']['mean']:.3f} [{s['auc']['min']:.3f}-{s['auc']['max']:.3f}] | "
              f"sens@10 {s['sens_primary']['mean']:.2f} [{s['sens_primary']['min']:.2f}-{s['sens_primary']['max']:.2f}] | "
              f"SCC->AK {s['scc_called_ak']['mean']:.1f} | SCC recall {s['scc_recall']['mean']:.2f} | "
              f"melanoma flagged {s['melanoma_patients_flagged']['mean']:.1f}")
    print("decision SCC boost:", res["decision_scc_boost"])
    print("decision ensemble :", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res["decision_ensemble"].items()})
    print("\nbaseline table (patient AUC | sens @10 slots, 10% prevalence | PPV):")
    rows = [("Arrival order (today)", None), ("Questionnaire", res["questionnaire"]), ("DINOv2 + logistic regression", res["dinov2_logreg"]),
            ("ConvNeXt, ISIC only", res["image_isic_only"]), ("ConvNeXt + PAD fine-tune (final)", res["final_image_model"]),
            ("Final image 60% + questionnaire 40%", res["fused"])]
    for name, r in rows:
        if r is None:
            print(f"  {name:<38} 0.500 | 0.10 | 0.10")
            continue
        p = primary(r)
        print(f"  {name:<38} {r['auc']:.3f} ({r['auc_ci'][0]:.3f}-{r['auc_ci'][1]:.3f}) | {p['sensitivity']:.2f} | {p['ppv']:.2f}")
    for k in ("image_minus_questionnaire", "fused_minus_questionnaire", "fused_minus_image"):
        print(f"  {k}: {res[k]['mean']:+.3f} ({res[k]['ci'][0]:+.3f} to {res[k]['ci'][1]:+.3f})")
    print("\ndiagnostics:", {k: round(v, 3) if isinstance(v, float) else v for k, v in res["diagnostics"].items()})
    print("MSK false alarms:", {k: round(v, 3) if isinstance(v, float) else v for k, v in res["msk_false_alarms"].items()})


if __name__ == "__main__":
    main()
