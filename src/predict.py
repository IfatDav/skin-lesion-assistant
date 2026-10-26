"""Inference entry point for the rest of the system (the agent's `predict_visual_risk` tool).

    from src.predict import VisualRiskModel
    model = VisualRiskModel()                       # loads the checkpoint(s) listed in models/final_model.json
    model.predict(["photo1.jpg", "photo2.jpg"])     # one patient, any number of photos

Returns class probabilities per photo and the patient-level scores, using the frozen aggregation rule
(max over photos). The scores rank patients; they are not calibrated probabilities of cancer.
"""
import json
from pathlib import Path

import torch
from PIL import Image

from src import config as C
from src.data import eval_transform
from src.model import create_model

MANIFEST = C.MODELS_DIR / "final_model.json"
CANCER_IDX = [C.CLASSES.index(c) for c in ("MEL", "SCC", "BCC")]
URGENT_IDX = [C.CLASSES.index(c) for c in ("MEL", "SCC")]


class VisualRiskModel:
    def __init__(self, manifest: Path = MANIFEST, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.models = []
        for name in json.loads(Path(manifest).read_text())["checkpoints"]:
            m = create_model(pretrained=False)
            m.load_state_dict(torch.load(C.MODELS_DIR / name, map_location="cpu")["state_dict"])
            self.models.append(m.to(self.device).eval())
        self.transform = eval_transform()

    @torch.no_grad()
    def predict(self, photos: list[str | Path]) -> dict:
        x = torch.stack([self.transform(Image.open(p).convert("RGB")) for p in photos]).to(self.device)
        probs = torch.stack([m(x).float().softmax(1) for m in self.models]).mean(0).cpu()
        per_photo = [{"photo": str(p), **{c: round(float(v), 4) for c, v in zip(C.CLASSES, row)}}
                     for p, row in zip(photos, probs)]
        cancer = probs[:, CANCER_IDX].sum(1)
        urgent = probs[:, URGENT_IDX].sum(1)
        top = int(cancer.argmax())
        return {"per_photo": per_photo,
                "cancer_score": round(float(cancer.max()), 4),
                "urgent_score": round(float(urgent.max()), 4),
                "most_suspicious_photo": str(photos[top]),
                "most_likely_class": C.CLASSES[int(probs[top].argmax())]}


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(VisualRiskModel().predict(sys.argv[1:]), indent=1))
