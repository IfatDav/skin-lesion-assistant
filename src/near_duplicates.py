"""Label-free audit: near-identical photographs that sit in different PAD splits (re-saved, re-cropped copies
that an exact file hash misses).

A 64-bit difference hash proposes candidates (distance <= 4). On smooth skin that hash is coarse and matches
unrelated photos, so each candidate is confirmed by pixel correlation on 128 px grayscale; only pairs above
0.95 count as duplicates.

Run:  python -m src.near_duplicates
"""
import json
from itertools import combinations

import numpy as np
from PIL import Image

from src import config as C
from src.data import load_index, resolve_path


def dhash(path, size=8) -> int:
    g = np.asarray(Image.open(path).convert("L").resize((size + 1, size), Image.LANCZOS), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def pixel_corr(path_a, path_b, n=128) -> float:
    def z(p):
        a = np.asarray(Image.open(p).convert("L").resize((n, n), Image.LANCZOS), dtype=np.float32)
        return (a - a.mean()) / (a.std() + 1e-6)
    return float((z(path_a) * z(path_b)).mean())


def main(max_dist: int = 4, min_corr: float = 0.95):
    df = load_index(["pad_train", "pad_val", "pad_test"], skip_duplicates=False)
    paths = [resolve_path(r) for _, r in df.iterrows()]
    h = np.array([dhash(p) for p in paths], dtype=np.uint64)
    splits, pats = df["split"].to_numpy(), df["patient_id"].to_numpy()
    candidates, confirmed = 0, []
    for i, j in combinations(range(len(h)), 2):
        if splits[i] == splits[j] or bin(int(h[i]) ^ int(h[j])).count("1") > max_dist:
            continue
        candidates += 1
        r = pixel_corr(paths[i], paths[j])
        if r >= min_corr:
            confirmed.append({"a": df.at[i, "image_id"], "b": df.at[j, "image_id"], "split_a": splits[i],
                              "split_b": splits[j], "same_patient": bool(pats[i] == pats[j]), "corr": round(r, 3)})
    out = {"images": int(len(df)), "hash_candidates": candidates, "min_corr": min_corr, "confirmed_pairs": confirmed}
    (C.REPORTS_DIR / "near_duplicates_pad.json").write_text(json.dumps(out, indent=1))
    print(f"{candidates} hash candidates across PAD splits, {len(confirmed)} confirmed (pixel corr >= {min_corr})")
    for p in confirmed:
        print(p)


if __name__ == "__main__":
    main()
