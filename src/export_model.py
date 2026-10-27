"""Packs the final model for sharing, after src.evaluate has chosen it.

  - image weights: models/<checkpoint>.safetensors (weights only, no code), attached to the GitHub release
    RELEASE_TAG and downloaded by src.predict when missing;
  - questionnaire: models/questionnaire.joblib, the fitted variant-D pipeline (a few KB, kept in git).
The manifest models/final_model.json gets the file names, sha256 hashes and the download address.

Run:  python -m src.export_model
"""
import hashlib
import json

import joblib
import torch
from safetensors.torch import save_file

from src import config as C
from src import questionnaire

RELEASE_TAG = "image-model-v1"
RELEASE_URL = f"https://github.com/IfatDav/skin-lesion-assistant/releases/download/{RELEASE_TAG}/"
MANIFEST = C.MODELS_DIR / "final_model.json"


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads(MANIFEST.read_text())
    weights = []
    for name in manifest["checkpoints"]:
        state = torch.load(C.MODELS_DIR / name, map_location="cpu")["state_dict"]
        out = C.MODELS_DIR / name.replace(".pt", ".safetensors")
        save_file({k: v.contiguous() for k, v in state.items()}, out)
        weights.append({"file": out.name, "sha256": sha256(out)})
        print(f"wrote {out.relative_to(C.REPO_ROOT)} ({out.stat().st_size / 1e6:.0f} MB)")
    joblib.dump(questionnaire.fit_model(), questionnaire.MODEL_FILE)
    print(f"wrote {questionnaire.MODEL_FILE.relative_to(C.REPO_ROOT)}")
    manifest.update({"weights": weights, "weights_url": RELEASE_URL,
                     "questionnaire_file": questionnaire.MODEL_FILE.name})
    MANIFEST.write_text(json.dumps(manifest, indent=1))
    print(f"upload the .safetensors file(s) to the GitHub release '{RELEASE_TAG}'")


if __name__ == "__main__":
    main()
