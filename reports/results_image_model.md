# Image model · results

Run on 26.10.2026, RTX 5070 Ti. **PAD test patients were never loaded.** All numbers are on the 220 PAD
validation patients (103 malignant) of the frozen split from notebook 02, at the charter's operating point, with
the same code as the questionnaire baseline, so every comparison is paired. Raw numbers: `final_results.json`;
figures: `figures/`; notebook: `notebooks/04_image_model.ipynb`.

## Pipeline

| Step | What | Data |
|---|---|---|
| Index | One row per image: labels, patient and lesion keys, file hash, split. Byte-identical duplicates removed before splitting | 8,837 images |
| Labels | 6 classes from the ISIC archive's fine labels: MEL, SCC (incl. keratoacanthoma, SCC in situ), BCC, AK, NEV, BKL. Cancer score = P(MEL)+P(SCC)+P(BCC) (charter M2) | |
| 1 · head warm-up | ConvNeXt-Tiny (`convnext_tiny.fb_in22k_ft_in1k`, 28M parameters), new 6-class head, body frozen, 2 epochs, lr 1e-3 | 4,325 ISIC clinical close-ups |
| 2 · full training | All layers, layer-wise lr decay 0.85 (lr 2e-4 at the head), cosine schedule, early stopping on ISIC validation | same |
| 3 · PAD fine-tuning | Stem and stages 0–1 frozen, lr 3e-5 (head 3e-4), SCC class weight ×2, early stopping on PAD validation patients | 1,430 photos of 878 PAD training patients |
| Fusion | 60% image rank + 40% questionnaire rank; nothing fitted | |

Common to all steps: 448 px, batch 32, bf16, AdamW (wd 0.05), cross-entropy with class weights 1/√n and label
smoothing 0.1, augmentation by rotation, flips, crop (70–100%), mild brightness/contrast/saturation, light blur.

## Decision rules, fixed before the runs

1. SCC weight ×2 is adopted only if, averaged over 3 seeds, fewer SCC photos are called AK and patient AUC drops
   by no more than 0.01. **Adopted**: SCC→AK 7.3 → 5.7, SCC recall 0.41 → 0.52, AUC 0.953 → 0.950.
2. The final model is the seed-42 run of the adopted variant, not the best seed.
3. Averaging the 3 seeds is adopted only if its paired AUC gain excludes zero. **Not adopted**: +0.0006
   (95% CI −0.007 to +0.008).
4. Fusion weights are fixed at 60/40.

## Baseline table · 220 PAD validation patients

Sensitivity and PPV at the primary operating point: 10 urgent slots per 100 arrivals, 10% malignant prevalence.

| Model | Patient AUC (95% CI) | Sensitivity | PPV |
|---|---|---|---|
| Arrival order (today) | 0.500 | 0.10 | 0.10 |
| Questionnaire only | 0.867 (0.815–0.913) | 0.52 | 0.53 |
| DINOv2 frozen + logistic regression | 0.880 (0.835–0.923) | 0.49 | 0.51 |
| ConvNeXt, ISIC only | 0.778 (0.719–0.836) | 0.30 | 0.30 |
| ConvNeXt + PAD fine-tuning (image part of the final model) | 0.956 (0.930–0.977) | 0.60 | 0.61 |
| **Final model: image 60% + questionnaire 40%** | **0.954 (0.928–0.978)** | **0.61** | **0.61** |

Paired AUC differences: image − questionnaire **+0.089** (+0.047 to +0.132); fused − questionnaire +0.087
(+0.054 to +0.124); fused − image −0.001 (−0.014 to +0.011).

At 15 slots the share of cancer patients on the list is 0.65 (questionnaire), 0.81 (image), 0.82 (fused).
Over three seeds the plain recipe gives AUC 0.950–0.959 and the SCC×2 recipe 0.947–0.956.

## Findings

1. **The image model ranks phone-photo patients better than the questionnaire**, by 9 AUC points, a paired and
   significant difference. At the primary point it puts 60% of cancer patients on the daily list (questionnaire 52%).
2. **The final model is the 60/40 fusion** (team decision, `decisions.md`). It ties the image model on these
   patients and keeps the answers in the score for poor photos. The weights were fixed before any result.
   **Unbiopsied lesions are easy cases:** in PAD they are all benign. On the 116 patients whose lesions were all
   biopsied (93 with cancer) the AUCs are 0.890 (final), 0.886 (image) and 0.846 (questionnaire); fused − image
   +0.005 (−0.024 to +0.033).
3. **Most of the clinic-to-phone gap was actinic keratosis.** Without the 81 patients who have an AK, the
   ISIC-only model already reaches 0.899. ISIC has 123 AK photos and PAD 730; the ISIC-only model calls AK cancer.
4. **The cheapest image baseline is already as good as the questionnaire** (0.880 vs 0.867) without training a
   single network weight. The fine-tuned network adds 7.6 AUC points over it.
5. **The gain holds under other aggregation rules:** 0.956 with max over photos, 0.937 with the mean,
   0.928 per photo.

## Leakage checks

- No patient and no byte-identical image shared between PAD train, validation and test (asserted in `build_index`).
- Near-identical photos across PAD splits: 95 hash candidates, 0 confirmed by pixel correlation.
- PAD split reproduced from notebook 02's code and checked against its 18 published counts; questionnaire baseline
  reproduced exactly (AUC 0.867, sensitivity 0.524, PPV 0.532).

## Risks and limitations

- **SCC → AK.** The final model still calls 8 of 34 SCC photos actinic keratosis, which the charter counts as
  non-cancer. The SCC weight helped (from 10) but did not solve it. AK can progress to SCC, so this is the hardest
  and most consequential boundary.
- **The SCC weight cost some melanoma on average:** melanoma patients on the list 2.7 → 2.0 of 6 over three
  seeds (the final model has 3 of 6). With 6 patients this is within noise, but the team should know.
- **Surgical-marker ink** is visible on some PAD photos and is a known shortcut. Unmeasured: an automatic detector
  failed validation. Decided: after the test run, test photos are labelled blind and benign inked photos are
  compared with benign photos without ink.
- **More photos for cancer patients in PAD** (1.93 vs 1.33 on average); the photo count alone reaches AUC 0.663,
  and the max rule turns it into extra chances. Patient-level numbers are therefore slightly optimistic.
- **Small validation set** (220 patients, 6 with melanoma), used to choose the epoch: expect the test score to be
  somewhat lower.
- **One programme, trained photographers.** Patients photographing themselves will produce worse photos.

## Not yet done

Decisions are frozen in `decisions.md`. In order: the team approves it, `python -m src.evaluate_test --frozen`
runs once, then the test photos are labelled for ink and `python -m src.ink_effect`.
