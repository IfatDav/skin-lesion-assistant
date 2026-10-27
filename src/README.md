# src · image model

Code for the image (CV) part of the triage pipeline. Images are not in the repo: the scripts read them from
`../DATA` next to the repo, or from the folder in the `SKIN_DATA_ROOT` environment variable. Results:
`reports/results_image_model.md` and `notebooks/04_image_model.ipynb`.

## Data layout

```
DATA/                                  # next to the repo, or set SKIN_DATA_ROOT
  metadata/PAD_UFES_20_metadata.csv
  isic_512*/isic_512/ISIC_*.jpg        # the Drive download; the timestamp in the folder name does not matter
  pad_original*/pad_original/PAT_*.png # PAD may come in two download parts
```

The index stores file names only, so it works on any machine with this layout. The ISIC labels
(`data/metadata/isic_clinical_metadata.csv`, pulled from the ISIC API) are in the repo.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Run order

```bash
python -m src.build_index --resize-pad            # index + splits, 512 px PAD copies              ~1 min
python -m src.train --seed 42                     # ISIC training -> models/convnext_milk_s42.pt  ~5 min
python -m src.finetune_pad --seed 42 --scc-boost 2   # PAD fine-tune -> models/convnext_pad_s42_scc2.pt ~1 min
python -m src.baseline_dinov2                     # cheapest image baseline                        ~1 min
python -m src.evaluate                            # decisions, final model, figures                ~5 min
python -m src.export_model                        # safetensors weights + questionnaire.joblib     <1 min
python -m src.near_duplicates                     # leakage audit across PAD splits                <1 min
```

`src.evaluate` expects seeds 42, 1 and 2 of `train` and of `finetune_pad` with `--scc-boost 1` and `2`.
**PAD test patients are never loaded by any of these scripts.** The one script that does,
`python -m src.evaluate_test --frozen`, runs once, after the team approves `reports/decisions.md`.

On Windows, run scripts as modules (as above), not from stdin or from notebook cells that define the dataset:
the data loader starts worker processes that must be able to import the code.

## Using the model (agent tool `predict_visual_risk`)

```python
from src.predict import VisualRiskModel
model = VisualRiskModel()                              # checkpoint(s) listed in models/final_model.json
model.predict(["photo1.jpg", "photo2.jpg"])            # one patient, any number of photos
# -> per-photo class probabilities, cancer_score, urgent_score, most_suspicious_photo, most_likely_class
```

The final model ranks a whole queue: 60% image rank + 40% questionnaire rank.

```python
from src.triage import Triage
Triage().rank([{"patient_id": "a", "photos": ["a1.jpg"], "age": 61, "region": "FACE", "itch": True,
                "grew": True, "hurt": False, "changed": False, "bleed": True, "elevation": True}, ...])
# -> the queue sorted by score, with the top 10 per 100 marked urgent
```

Scores rank patients; they are not calibrated probabilities.

**Weights.** The image weights (111 MB, `.safetensors`) are too big for git. They are attached to the GitHub
release `image-model-v1`, and `src.predict` downloads them into `models/` on first use and checks their sha256
against `models/final_model.json`. The questionnaire model is in git as `models/questionnaire.joblib`; it needs
the scikit-learn version pinned in `requirements.txt`.

## Files

| File | Purpose |
|---|---|
| `config.py` | paths, the 6 classes, label mappings, urgency tiers |
| `build_index.py` | `data/splits/image_index.csv`: one row per image with labels, keys, hash and split |
| `pad_split.py` | the team's frozen PAD patient split (notebook 02), reproduced and verified against its counts |
| `data.py` | dataset, augmentation, loaders |
| `model.py` | ConvNeXt-Tiny with a 6-class head; freezing and optimiser rules per step |
| `train.py` | ISIC training (head warm-up, then all layers) |
| `finetune_pad.py` | PAD fine-tuning (late layers only) |
| `metrics.py`, `operating_point.py` | cancer score, patient aggregation, the charter's slot × prevalence operating point |
| `questionnaire.py` | the questionnaire baseline from notebook 02, reproduced for paired comparison |
| `fusion.py` | 60/40 rank-average fusion |
| `triage.py` | the final model: ranks a queue from photos and answers |
| `baseline_dinov2.py` | frozen DINOv2 features + logistic regression |
| `evaluate.py` | applies the pre-set decision rules, picks the final model, writes results and figures |
| `evaluate_test.py` | the single run on PAD test patients, written before it runs |
| `near_duplicates.py` | near-identical photos across PAD splits |
| `ink_effect.py` | analyses the manual ink labels in `evaluation/` |
| `export_model.py` | packs the final model: `.safetensors` weights for the release, `questionnaire.joblib` |
| `predict.py` | inference entry point for the agent; downloads the weights on first use |

## Index splits

| split | images | use |
|---|---|---|
| `milk_train` / `milk_val` | 4,325 / 763 | ISIC training and checkpoint selection; split by lesion |
| `pad_train` / `pad_val` / `pad_test` | 1,440 / 357 / 501 | the team's frozen PAD patient split; if the original `pad_split.csv` is placed in `data/splits/`, it is used instead |
| `msk_eval` | 1,241 | Memorial Sloan Kettering: 1,234 benign from 73 patients; kept out of training |
| `excluded` | 210 | 129 rare classes, 79 without a fine label, 2 byte-identical duplicates |

## Notes

- Labels come from the ISIC archive's `diagnosis_3`, not the MILK10k challenge CSV, whose `AKIEC` column merges
  actinic keratosis (pre-cancer) with SCC in situ (cancer).
- `timm`'s `layer_decay` only stores an `lr_scale` per parameter group; `model.make_optimizer` applies it.
- PAD lesion ids are reused across patients (250 of 1,641), so the lesion key is `patient_id::lesion_id`.
