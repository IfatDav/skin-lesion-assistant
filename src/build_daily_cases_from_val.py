
"""Build a reproducible validation-only model-score-to-agent demonstration.

This module belongs in src/build_daily_cases_from_val.py.
Run the following commands from the repository root:

    python -m src.build_daily_cases_from_val

    # After reviewing the ten real validation photographs and recording quality:
    python -m src.build_daily_cases_from_val --quality-file data/processed/val_top10_quality_template.csv

    # Optional fresh image-model inference: first generate its list and template,
    # then review the selected photos and rebuild with their quality annotations.
    python -m src.build_daily_cases_from_val --fresh-inference
    python -m src.build_daily_cases_from_val --fresh-inference --quality-file data/processed/val_top10_quality_fresh_template.csv

By default, scores come from previously run PAD validation models (no new inference).
With --fresh-inference, the script re-runs both trained models on PAD photos.
We restrict this demo to patients with exactly one lesion and one photo, so the
patient-level score, displayed photo, and symptom form refer to the same lesion.
This restricts 220 validation patients (103 malignant, 46.8%) to 138 eligible
patients (47 malignant, 34.1%). Controlled 10% prevalence in the 100-person
demo does not remove selection bias: this is NOT a representative clinical test.
The 10% malignancy prevalence is achieved by controlled sampling for a simulated
day, NEVER by using diagnosis during the ranking stage or agent review.
The scores and percentile bands are relative ranking values, NOT probabilities.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.fusion import fuse, rank_band

ROOT = Path(__file__).resolve().parents[1]  # repository root; script lives in src/
SCORES_CSV = ROOT / "data/processed/val_scores_fused.csv"
FEATURES_CSV = ROOT / "data/processed/pad_symptom_features.csv"
IMAGES_CSV = ROOT / "data/splits/pad_split.csv"
OUT = ROOT / "data/processed/daily_cases_val.csv"
AUDIT = ROOT / "data/processed/val_100_candidates_audit.csv"
QUALITY_TEMPLATE = ROOT / "data/processed/val_top10_quality_template.csv"

SEED = 42
N_CANDIDATES = 100
N_MALIGNANT = 10
N_PRIORITY = 10
W_IMAGE = 0.6  # locked final model in models/final_model.json
SYMPTOMS = ("itch", "grew", "hurt", "changed", "bleed", "elevation")
VALID_QUALITY = {"usable", "borderline", "unusable", "not_a_lesion"}


def yes_no(value) -> str:
    if pd.isna(value):
        return "unknown"
    s = str(value).strip().lower()
    if s in ("true", "1", "yes"):
        return "yes"
    if s in ("false", "0", "no"):
        return "no"
    return "unknown"


def add_fresh_model_scores(day: pd.DataFrame) -> pd.DataFrame:
    """Optional real photo -> image/questionnaire model inference -> fusion.

    Requires the original PAD photos (or 512px cache), image_index.csv, the
    released safetensors image weights, questionnaire.joblib, and PyTorch/timm.
    Validation patients only; no training or test labels are used at inference.
    """
    from src import config as C
    from src.data import resolve_path
    from src.triage import Triage

    idx = pd.read_csv(C.INDEX_CSV, dtype=str).set_index("image_id")
    missing = sorted(set(day.img_id) - set(idx.index))
    if missing:
        raise ValueError(f"Images missing from image_index.csv: {missing[:3]}")
    patient_inputs = []
    for r in day.to_dict("records"):
        row = idx.loc[r["img_id"]]
        if row["source"] != "pad" or row["split"] != "pad_val":
            raise ValueError(f"Not a PAD validation photo: {r['img_id']}")
        path = resolve_path(row)
        if not path.is_file():
            raise FileNotFoundError(f"PAD photo not downloaded: {path}")
        inp = {"patient_id": r["patient_id"], "photos": [str(path)],
               "age": int(r["age"]), "region": r["region"]}
        inp.update({sym: yes_no(r[sym]) == "yes" if yes_no(r[sym]) != "unknown" else None
                    for sym in SYMPTOMS})
        patient_inputs.append(inp)

    print(f"Running fresh model inference on {len(patient_inputs)} PAD validation photographs...")
    ranked = Triage().rank(patient_inputs)
    assert len(ranked) == len(day) and ranked.index.is_unique
    day = day.set_index("patient_id")
    day["image_score"] = ranked["image_score"].reindex(day.index)
    day["questionnaire_score"] = ranked["questionnaire_score"].reindex(day.index)
    # Triage already computes the same 60/40 percentile-rank fusion below.
    day = day.reset_index()
    return day


def build(quality_file: Path | None = None, fresh_inference: bool = False) -> pd.DataFrame:
    scores = pd.read_csv(SCORES_CSV)
    features = pd.read_csv(FEATURES_CSV)
    images = pd.read_csv(IMAGES_CSV)

    assert scores.patient_id.is_unique, "Expected one precomputed score row per patient"
    assert set(scores.columns) >= {"patient_id", "y", "image_score", "questionnaire_score"}
    features = features.loc[features["split"].eq("val")].copy()
    images = images.loc[images["split"].eq("val")].copy()

    # Match one patient, one lesion, one photo, for a clean and honest demo.
    # This intentionally omits multi-lesion cases: their max photo score and max
    # questionnaire score can come from different lesions.
    one_lesion = features.groupby("patient_id").filter(lambda x: len(x) == 1)
    one_photo = images.groupby("patient_id").filter(lambda x: len(x) == 1)
    merged = (scores.merge(one_lesion, on="patient_id", how="inner", validate="one_to_one")
                    .merge(one_photo[["patient_id", "lesion_key", "img_id"]],
                           on=["patient_id", "lesion_key"], how="inner", validate="one_to_one"))
    assert merged.patient_id.is_unique
    assert (merged["y"].astype(int) == merged["label"].astype(int)).all(), "Mismatched outcome labels"
    assert (merged["split"] == "val").all(), "Non-validation patient included"
    print(f"Demo eligibility: {len(scores)} validation patients "
          f"({scores['y'].astype(int).mean():.1%} malignant) -> "
          f"{len(merged)} single-lesion/single-photo patients "
          f"({merged['y'].astype(int).mean():.1%} malignant). "
          "This is a non-representative demonstration subset.")

    pos, neg = merged.loc[merged.y.eq(1)], merged.loc[merged.y.eq(0)]
    if len(pos) < N_MALIGNANT or len(neg) < (N_CANDIDATES - N_MALIGNANT):
        raise ValueError(f"Not enough single-lesion/single-photo patients: {len(pos)} positive, {len(neg)} negative")

    # Controlled scenario sampling: labels are used ONLY to create 10% prevalence.
    # The ranking and the agent DO NOT receive the labels.
    day = pd.concat([
        pos.sample(n=N_MALIGNANT, random_state=SEED),
        neg.sample(n=N_CANDIDATES - N_MALIGNANT, random_state=SEED + 1),
    ], ignore_index=True)
    assert day.patient_id.is_unique
    if fresh_inference:
        day = add_fresh_model_scores(day)

    img = day.set_index("patient_id")["image_score"]
    questionnaire = day.set_index("patient_id")["questionnaire_score"]
    fused = fuse(img, questionnaire, w_image=W_IMAGE)
    day = day.set_index("patient_id")
    day["final_score"] = fused
    day["img_pct"] = img.rank(pct=True)
    day["questionnaire_pct"] = questionnaire.rank(pct=True)
    day = day.reset_index().sort_values(["final_score", "patient_id"], ascending=[False, True])
    day["rank"] = range(1, N_CANDIDATES + 1)
    day["urgent"] = day["rank"] <= N_PRIORITY

    out_file = (ROOT / "data/processed/daily_cases_val_fresh.csv") if fresh_inference else OUT
    audit_file = (ROOT / "data/processed/val_100_candidates_fresh_audit.csv") if fresh_inference else AUDIT
    quality_template = ((ROOT / "data/processed/val_top10_quality_fresh_template.csv")
                        if fresh_inference else QUALITY_TEMPLATE)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    # This audit is for the project team only; do not give labels to the agent.
    day[["patient_id", "img_id", "y", "image_score", "questionnaire_score",
         "final_score", "rank", "urgent"]].to_csv(audit_file, index=False)

    top = day.head(N_PRIORITY).copy()
    template = top[["patient_id", "img_id"]].rename(columns={"img_id": "image_filename"})
    template["image_quality"] = ""
    template["image_issues"] = ""
    # Do not destroy a manually annotated template on a later run.
    if not quality_template.exists():
        template.to_csv(quality_template, index=False)

    if quality_file is None:
        qualities = {pid: {"image_quality": "not_checked", "image_issues": "quality not reviewed"}
                     for pid in top.patient_id}
    else:
        q = pd.read_csv(quality_file).fillna("")
        if not {"patient_id", "image_quality", "image_issues"} <= set(q.columns):
            raise ValueError("Quality CSV needs patient_id, image_quality, image_issues columns")
        if not q.patient_id.is_unique or set(q.patient_id) != set(top.patient_id):
            raise ValueError("Quality file must contain exactly the ten selected patient IDs")
        bad = q.loc[~q.image_quality.isin(VALID_QUALITY)]
        if len(bad):
            raise ValueError(f"Missing/invalid manual image quality labels for: {bad.patient_id.tolist()}")
        qualities = q.set_index("patient_id")[["image_quality", "image_issues"]].to_dict("index")

    records = []
    for r in top.to_dict("records"):
        quality = qualities[r["patient_id"]]
        record = {
            "patient_id": r["patient_id"],
            "rank": int(r["rank"]),
            "risk_score": round(float(r["final_score"]), 4),
            "risk_band": rank_band(float(r["final_score"])),
            "image_risk_band": rank_band(float(r["img_pct"])),
            "symptom_risk_band": rank_band(float(r["questionnaire_pct"])),
            "age": int(r["age"]) if pd.notna(r["age"]) else "",
            "body_site": str(r["region"]).lower() if pd.notna(r["region"]) else "",
            "patient_note": "",  # PAD has no free-text notes
            "image_path": str(r["img_id"]),  # filename; agent does not load pixels
            "image_quality": quality["image_quality"],
            "image_issues": quality["image_issues"],
        }
        record.update({s: yes_no(r[s]) for s in SYMPTOMS})
        records.append(record)

    out = pd.DataFrame.from_records(records)
    assert len(out) == N_PRIORITY and out.patient_id.is_unique
    assert out["rank"].tolist() == list(range(1, N_PRIORITY + 1))
    assert set(out.columns) >= {"patient_id", "risk_score", "risk_band", "image_risk_band",
                                 "symptom_risk_band", "body_site", "image_quality", *SYMPTOMS}
    out.to_csv(out_file, index=False)
    print(f"Validation-only scenario: {N_CANDIDATES} distinct candidates; {N_MALIGNANT} malignant by simulation design")
    print("Model score source:", "fresh PAD photo inference" if fresh_inference else "stored validation predictions")
    print(f"Selected {N_PRIORITY} priority cases (labels NEVER exported to agent file)")
    print(f"Daily agent input: {out_file.relative_to(ROOT)}")
    print(f"Candidate audit:    {audit_file.relative_to(ROOT)}")
    print(f"Quality template:   {quality_template.relative_to(ROOT)}")
    if quality_file is None:
        print("WARNING: image_quality is 'not_checked'. For a smoke test, set "
              "CASES_FILE to the generated CSV and USE_AGENT = False in notebook 09 "
              "(or use --no-agent with agent/triage_agent.py). Review all 10 "
              "photos and re-run this script with --quality-file before an agent run.")
    else:
        print("Manual image-quality labels loaded. Ready for an agent integration test (not clinical use).")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality-file", type=Path, default=None,
                    help="CSV of ten manually reviewed image_quality statuses")
    ap.add_argument("--fresh-inference", action="store_true",
                    help="Recompute scores from PAD photographs using both trained models (requires image files)")
    args = ap.parse_args()
    build(args.quality_file, fresh_inference=args.fresh_inference)

