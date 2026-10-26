"""Build data/splits/image_index.csv: one row per image with labels, group keys and split.

Run:  python -m src.build_index [--resize-pad]
"""
import argparse
import hashlib
import random
from pathlib import Path

import pandas as pd
from PIL import Image

from src import config as C
from src.pad_split import patient_split


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def isic_rows() -> pd.DataFrame:
    m = pd.read_csv(C.META_DIR / "isic_clinical_metadata.csv", dtype=str).fillna("")
    src = m["attribution"].map(lambda a: "milk" if a == "MILK study team" else "msk" if a == C.MSK_ATTRIBUTION else "isic_other")
    df = pd.DataFrame({
        "image_id": m["isic_id"],
        "image_path": [str((C.ISIC_DIR / f"{i}.jpg").relative_to(C.DATA_ROOT)) for i in m["isic_id"]],
        "source": src,
        "patient_id": m["patient_id"],
        "lesion_key": m["lesion_id"].where(m["lesion_id"] != "", m["isic_id"]),
        "label_raw": m["diagnosis_3"].where(m["diagnosis_3"] != "", m["diagnosis_1"]),
        "label6": m["diagnosis_3"].map(C.ISIC_TO_CLASS).fillna(""),
        "diagnosis_1": m["diagnosis_1"],
        "age": m["age_approx"], "sex": m["sex"], "site": m["anatom_site_1"],
        "fitzpatrick": m["fitzpatrick_skin_type"],
    })
    return df


def pad_rows() -> pd.DataFrame:
    m = pd.read_csv(C.META_DIR / "PAD_UFES_20_metadata.csv", dtype=str).fillna("")
    where = {p.name: p for d in C.PAD_DIRS for p in d.iterdir()}
    df = pd.DataFrame({
        "image_id": m["img_id"],
        "image_path": [str(where[i].relative_to(C.DATA_ROOT)) for i in m["img_id"]],
        "source": "pad",
        "patient_id": m["patient_id"],
        # Raw PAD lesion ids are reused across patients (250 of 1,641), so the key needs both.
        "lesion_key": m["patient_id"] + "::" + m["lesion_id"],
        "label_raw": m["diagnostic"],
        "label6": m["diagnostic"].map(C.PAD_TO_CLASS),
        "diagnosis_1": "",
        "age": m["age"], "sex": m["gender"], "site": m["region"],
        "fitzpatrick": m["fitspatrick"].replace("UNK", ""),
    })
    return df


def cancer_label(row) -> str:
    if row.label6:
        return "1" if row.label6 in C.CANCER else "0"
    return {"Malignant": "1", "Benign": "0"}.get(row.diagnosis_1, "")


def assign_splits(df: pd.DataFrame) -> pd.DataFrame:
    df["split"], df["exclude_reason"], df["duplicate_of"] = "", "", ""

    # Exact byte duplicates: keep the first id, point the others at it.
    first = {}
    for i in df.sort_values("image_id").index:
        h = df.at[i, "md5"]
        if h in first:
            df.at[i, "duplicate_of"] = df.at[first[h], "image_id"]
        else:
            first[h] = i

    isic_train_pool = df["source"].isin(["milk", "isic_other"])
    df.loc[df["source"] == "msk", "split"] = "msk_eval"
    pad_meta = pd.read_csv(C.META_DIR / "PAD_UFES_20_metadata.csv", dtype=str)
    pad_map = patient_split(pad_meta)
    is_pad = df["source"] == "pad"
    df.loc[is_pad, "split"] = "pad_" + df.loc[is_pad, "patient_id"].map(pad_map)

    no_fine = isic_train_pool & (df["label_raw"] == df["diagnosis_1"])
    other = isic_train_pool & ~no_fine & (df["label6"] == "")
    dup = isic_train_pool & (df["duplicate_of"] != "")
    df.loc[no_fine, ["split", "exclude_reason"]] = ["excluded", "no_fine_label"]
    df.loc[other, ["split", "exclude_reason"]] = ["excluded", "rare_class"]
    df.loc[dup & (df["split"] == ""), ["split", "exclude_reason"]] = ["excluded", "duplicate"]

    eligible = isic_train_pool & (df["split"] == "")
    lesions = sorted(df.loc[eligible, "lesion_key"].unique())
    random.Random(C.SEED).shuffle(lesions)
    val = set(lesions[: round(len(lesions) * 0.15)])
    df.loc[eligible, "split"] = df.loc[eligible, "lesion_key"].map(lambda k: "milk_val" if k in val else "milk_train")
    return df


def resize_pad(df: pd.DataFrame) -> None:
    C.PAD_RESIZED_DIR.mkdir(parents=True, exist_ok=True)
    for rel in df.loc[df["source"] == "pad", "image_path"]:
        out = C.PAD_RESIZED_DIR / (Path(rel).stem + ".jpg")
        if out.exists():
            continue
        im = Image.open(C.DATA_ROOT / rel).convert("RGB")
        im.thumbnail((512, 512), Image.LANCZOS)
        im.save(out, quality=95)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resize-pad", action="store_true", help="write 512px copies of PAD to data/processed/pad_512")
    args = ap.parse_args()

    df = pd.concat([isic_rows(), pad_rows()], ignore_index=True)
    df["md5"] = [md5(C.DATA_ROOT / p) for p in df["image_path"]]
    df["cancer"] = df.apply(cancer_label, axis=1)
    df["tier"] = df["label6"].map(C.TIER).fillna("")
    df = assign_splits(df)

    C.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["image_id", "image_path", "source", "split", "exclude_reason", "duplicate_of", "patient_id", "lesion_key",
            "label_raw", "label6", "tier", "cancer", "age", "sex", "site", "fitzpatrick", "md5"]
    df[cols].to_csv(C.INDEX_CSV, index=False)

    print(f"wrote {C.INDEX_CSV.relative_to(C.REPO_ROOT)}: {len(df)} images")
    print(pd.crosstab(df["split"], df["label6"].replace("", "-"), margins=True).to_string())
    print("\nexcluded:", df.loc[df["split"] == "excluded", "exclude_reason"].value_counts().to_dict())
    print("duplicate copies:", (df["duplicate_of"] != "").groupby(df["source"]).sum().to_dict())
    tr, va = df[df["split"] == "milk_train"], df[df["split"] == "milk_val"]
    assert not set(tr["lesion_key"]) & set(va["lesion_key"]), "lesion leak between milk_train and milk_val"
    assert not set(tr["md5"]) & set(va["md5"]), "identical image in milk_train and milk_val"
    print("leak checks passed: no shared lesion and no identical image between milk_train and milk_val")
    pad = df[df["source"] == "pad"]
    for a, b in [("pad_train", "pad_val"), ("pad_train", "pad_test"), ("pad_val", "pad_test")]:
        pa, pb = pad[pad["split"] == a], pad[pad["split"] == b]
        assert not set(pa["patient_id"]) & set(pb["patient_id"]), f"patient in both {a} and {b}"
        assert not set(pa["md5"]) & set(pb["md5"]), f"identical image in both {a} and {b}"
    print("PAD split matches notebook 02 and has no patient or identical image across train/val/test")

    if args.resize_pad:
        resize_pad(df)
        print(f"PAD resized to {C.PAD_RESIZED_DIR.relative_to(C.REPO_ROOT)}")


if __name__ == "__main__":
    main()
