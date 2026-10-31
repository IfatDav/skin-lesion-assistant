# src · image model

Code for the image (CV) part of the triage pipeline. Images are not in the repo: the scripts read them from
`../DATA` next to the repo, or from the folder in the `SKIN_DATA_ROOT` environment variable. Results:
`reports/results_image_model.md` and `notebooks/05_image_model.ipynb`.

## Data layout

```
DATA/                                  # next to the repo, or set SKIN_DATA_ROOT
  metadata/PAD_UFES_20_metadata.csv
  isic_512*/isic_512/ISIC_*.jpg        # the Drive download; the timestamp in the folder name does not matter
  pad_original*/pad_original/PAT_*.png # PAD may come in two download parts
```

The index stores file names only, so it works on any machine with this layout. The ISIC labels
(`data/metadata/isic_pool_api_metadata.csv`, pulled from the ISIC API) are in the repo. The notebooks'
own ISIC table, with the derived exclusion columns, is a separate file: `data/raw/isic_clinical_metadata.csv`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```


## Run order

The full training and validation reproduction requires three random seeds (42, 1 and 2). For each seed, the ISIC model is trained once and then fine-tuned on PAD in two variants: standard class weighting (`--scc-boost 1`) and doubled SCC class weighting (`--scc-boost 2`).

Run the following commands from the repository root on a CUDA-enabled machine with the required image files available.

```bash
# 1. Build the index and prepare PAD images
python -m src.build_index --resize-pad

# 2. Train the ISIC image model with three seeds
python -m src.train --seed 42
python -m src.train --seed 1
python -m src.train --seed 2

# 3. Fine-tune both PAD variants for every seed
python -m src.finetune_pad --seed 42 --scc-boost 1
python -m src.finetune_pad --seed 42 --scc-boost 2

python -m src.finetune_pad --seed 1 --scc-boost 1
python -m src.finetune_pad --seed 1 --scc-boost 2

python -m src.finetune_pad --seed 2 --scc-boost 1
python -m src.finetune_pad --seed 2 --scc-boost 2

# 4. Evaluate the baselines and select the final model
python -m src.baseline_dinov2
python -m src.evaluate

# 5. Export the selected model
python -m src.export_model

# 6. Run the cross-split near-duplicate audit
python -m src.near_duplicates
```

`src.evaluate` requires all six PAD fine-tuning checkpoints and the three corresponding ISIC training checkpoints. It applies the frozen validation decision rules, compares the SCC-weighting variants and the seed ensemble, and writes the selected model manifest to `models/final_model.json`.

**Held-out test separation.** The training, model-selection and validation scripts do not use held-out PAD test outcomes to select the model. `src.build_index` does process test metadata to construct and verify the frozen splits, while `src.near_duplicates` reads test photographs only for a label-free cross-split leakage audit.

The held-out test was executed once on 30 October 2026 using `python -m src.evaluate_test --frozen`, following the decisions in `reports/decisions.md`. Its results are preserved in `reports/test_results.json`. The test command is not part of routine model selection or validation reproduction.

On Windows, run scripts as modules (as above), not from stdin or notebook cells that define the dataset: data-loader worker processes must be able to import the code.


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


## Validation-to-Agent integration (CP3)

`src/build_daily_cases_from_val.py` provides the reproducible connection between PAD validation predictions and the Agent's daily patient list.

By default, the script uses previously computed validation predictions rather than running fresh image inference.

### Demonstration population

The starting point is 220 PAD validation patients, including 103 with malignant lesions.

To ensure that the photograph, symptom answers and prediction correspond to the same lesion, the demonstration includes only patients with exactly one lesion and one photograph. This leaves 138 eligible patients, including 47 with malignant lesions.

A seeded simulation selects 100 candidates, including exactly 10 malignant cases by design. The frozen 60% image / 40% questionnaire percentile-rank fusion ranks these patients and selects the 10 highest-priority cases.

Six of the ten malignant candidates were selected for priority review.

This controlled sampling is intended for technical integration, not for estimating real-world sensitivity or clinical prevalence.

### Reproduce the saved-prediction demonstration

The required processed validation inputs are already committed in the repository:

- `data/processed/val_scores_fused.csv` — stored model predictions.
- `data/processed/pad_symptom_features.csv` — symptom features.
- `data/splits/pad_split.csv` — frozen patient and photograph mapping.
- `data/processed/val_top10_quality_template.csv` — saved image-quality annotations.

Run from the repository root:

```bash
python -m src.build_daily_cases_from_val --quality-file data/processed/val_top10_quality_template.csv
```

The script produces:

- `data/processed/daily_cases_val.csv` — ranked Agent input, without diagnostic labels.
- `data/processed/val_100_candidates_audit.csv` — evaluation audit containing ground-truth labels.
- `data/processed/val_top10_quality_template.csv` — image-quality annotation template, preserved if already present.

**Do not provide the diagnostic audit file to the Agent.**

The saved image-quality annotations were produced through AI-assisted visual review of screenshots: eight images were marked `usable`, and two were marked `borderline` with `glare`. These labels are demonstration inputs, not results from a validated image-quality classifier or independent dermatologist assessment.

### Generate a report without the Agent

For a deterministic smoke test using the ranked cases:

```bash
python agent/triage_agent.py --cases data/processed/daily_cases_val.csv --no-agent --out reports/agent/validation_smoke.docx
```

The report-generation command requires the Agent-layer Python dependencies described in `agent/triage_agent.py`.

The `--no-agent` option generates a report from fixed model evidence. It does not execute an LLM Agent.

### Run the actual Agent

The completed Agent integration is documented in `notebooks/09_agent_daily_report.ipynb`.

Set:

```python
CASES_FILE = os.path.join(PROCESSED, "daily_cases_val.csv")
USE_AGENT = True
```

The executed notebook demonstrates 10 successful Agent reviews out of 10 patients, with zero retries, zero fallbacks and no ranking changes.

The committed outputs are:

- `reports/agent/triage_report_validation_20261031.docx`
- `reports/agent/triage_report_validation_20261031_agent_log.json`

This establishes technical integration between stored validation predictions, prioritization, the Agent and the final Word report. It does not establish clinical accuracy or deployment readiness.

### Optional fresh inference

The builder also supports:

```bash
python -m src.build_daily_cases_from_val --fresh-inference
```

This option requires the original PAD validation photographs, model weights and dependencies. It generates separate `_fresh` outputs and a new quality-review template.

The committed CP3 integration demonstration used stored predictions, not this optional fresh-inference path.


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
| `build_daily_cases_from_val.py` | validation-only bridge from stored model predictions to ranked Agent input; supports optional fresh inference and image-quality annotations |
| `baseline_dinov2.py` | frozen DINOv2 features + logistic regression |
| `evaluate.py` | applies the pre-set decision rules, picks the final model, writes results and figures |
| `evaluate_test.py` | the single run on PAD test patients, written before it runs |
| `near_duplicates.py` | near-identical photos across PAD splits |
| `ink_effect.py` | analyses the manual ink labels in `evaluation/` |
| `export_model.py` | packs the final model: `.safetensors` weights for the release, `questionnaire.joblib` |
| `predict.py` | inference entry point for the agent; downloads the weights on first use |

`reports/image_model_report.html` is a standalone HTML summary of the validation run, written by hand from
`final_results.json`; nothing reads it.

## Index splits

| split | images | use |
|---|---|---|
| `milk_train` / `milk_val` | 4,325 / 763 | ISIC training and checkpoint selection; split by lesion |
| `pad_train` / `pad_val` / `pad_test` | 1,440 / 357 / 501 | the team's frozen PAD patient split; if the original `pad_split.csv` is placed in `data/splits/`, it is used instead |
| `msk_eval` | 1,241 | Memorial Sloan Kettering: 1,234 benign from 73 patients; kept out of training |
| `excluded` | 210 | 129 rare classes, 79 without a fine label, 2 duplicate photographs (byte-identical once resized to 512 px, and an identical perceptual hash; MD5 over the originals does not see them) |

## Notes

- Labels come from the ISIC archive's `diagnosis_3`, not the MILK10k challenge CSV, whose `AKIEC` column merges
  actinic keratosis (pre-cancer) with SCC in situ (cancer).
- `timm`'s `layer_decay` only stores an `lr_scale` per parameter group; `model.make_optimizer` applies it.
- PAD lesion ids are reused across patients (250 of 1,641), so the lesion key is `patient_id::lesion_id`.
