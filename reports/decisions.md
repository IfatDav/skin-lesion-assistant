# Image model · frozen decisions

Agreed before the single PAD test run. Once `python -m src.evaluate_test --frozen` has run, nothing here
changes; anything learned afterwards goes into the report as a finding, not into the model.

## Model

| # | Decision | Reason |
|---|---|---|
| 1 | **Final model: 60% image rank + 40% questionnaire rank.** Image alone and questionnaire alone are reported for comparison only. | On validation the fusion ties the image model (AUC 0.954 vs 0.956, difference −0.001, CI −0.014 to +0.011) and keeps the answers in the score for cases where the photo is poor. The weights were fixed before any result, so nothing was fitted on validation. |
| 2 | **Six classes** (MEL, SCC, BCC, AK, NEV, BKL), mapped to one score. | Keeps pre-cancer (AK) apart from cancer; matches charter M2. |
| 3 | **Ranking score: all cancers**, P(MEL)+P(SCC)+P(BCC), max over a patient's photos. The urgent score P(MEL)+P(SCC) is shown as a secondary measure. | Charter M2. BCC is most of the cancers in PAD. |
| 4 | **Operating point: 10 urgent slots per 100 arrivals at 10% prevalence**; 5 and 15 slots are reported too. | Notebook 02's definition. |
| 5 | Image model settings follow the pre-set rules in `src/evaluate.py`: SCC weight ×2 adopted, seed-42 run, no seed averaging. | Rules written before the runs. |

## Data

| # | Decision | Reason |
|---|---|---|
| 6 | Keep the 123 ISIC actinic keratosis photos in training. | Without them the model barely sees AK in clinic photos. |
| 7 | MSK only for the false-alarm check, not for training. | No full six-class labels. |
| 8 | No HAM10000. | Overlaps ISIC; avoids duplicates across training and evaluation. |

## Known limits, stated before the test run

- **Surgical-marker ink is not measured yet.** After the test run, the test photos are labelled for ink
  without seeing diagnosis or score (`evaluation/ink_labeling_test.csv`, written by the test script), and
  `python -m src.ink_effect` checks whether **benign** inked photos score
  higher than benign photos without ink. Removing inked photos from training was rejected: it would need all
  1,440 training photos labelled, would remove mostly biopsied (cancer) photos, and would need retraining.
- **Unbiopsied lesions are easy cases.** In PAD every cancer was biopsied; the 566 development lesions without
  a biopsy are all benign, diagnosed by eye, and also the ones missing the history block of the questionnaire.
  They inflate every model. On validation patients whose lesions were all biopsied (116, 93 with cancer):
  image 0.886, questionnaire 0.846, fusion 0.890. The test run reports both groups.
- **Validation numbers are slightly optimistic**: epochs, the SCC weight and the seed were chosen on them.
  The test run is the clean estimate.
- The questionnaire model (variant D) uses age, region and six symptoms only, not the history block, so the
  block's missingness does not enter the score.

## Order

1. ~~Team approves this file.~~ **Approved 26.10.2026.**
2. `python -m src.evaluate_test --frozen` — once.
3. Ink labelling of the test photos (`evaluation/INK_LABELING.md`), then `python -m src.ink_effect`.
