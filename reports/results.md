# Results

**Every number here is on validation data. The held-out PAD test patients have not been opened.**
They are opened once, after the team approves `reports/decisions.md`, and the test figures replace this
sentence rather than these numbers.

All model rows are on the same **220 PAD validation patients** (103 with a malignant lesion) from the
frozen patient split created in notebook 02, aggregated image → lesion → patient by maximum, at the
charter's primary operating point: **10 urgent slots per 100 daily arrivals, 10% malignant prevalence.**

---

## 1 · The baseline table

| Model | Patient AUC (95% CI) | Sensitivity | PPV | What it costs to build |
|---|---|---|---|---|
| Arrival order — today's practice | 0.500 | 0.10 | 0.10 | nothing |
| **Questionnaire only — the baseline** | **0.867** (0.815–0.913) | **0.52** | **0.53** | seconds, and no photograph |
| DINOv2 frozen + logistic regression | 0.880 (0.835–0.923) | 0.49 | 0.51 | one forward pass, no training |
| ConvNeXt, ISIC only | 0.778 (0.719–0.836) | 0.30 | 0.30 | ~1 GPU-hour |
| ConvNeXt + PAD fine-tuning | 0.956 (0.930–0.977) | 0.60 | 0.61 | ~2 GPU-hours |
| **Final: image 60% + questionnaire 40%** | **0.954** (0.928–0.978) | **0.61** | **0.61** | the above; the fusion fits nothing |

Paired differences, bootstrapped on the same patients:

| | ΔAUC | 95% CI |
|---|---|---|
| image − questionnaire | **+0.089** | +0.047 to +0.132 |
| fused − questionnaire | +0.087 | +0.054 to +0.124 |
| fused − image | −0.001 | −0.014 to +0.011 |

**What the table is for.** The questionnaire reaches 0.867 without a photograph at all, and a frozen
DINOv2 with a logistic regression on top matches it without training a single weight. Those two rows are
the honest competition. The fine-tuned network earns its place by +0.089 AUC and by putting 60% of cancer
patients on the daily list instead of 52% — a paired, significant difference on the same patients.

**The fusion does not beat the image model.** The difference is −0.001 and the interval spans zero. It was
chosen in advance, before any result, because it keeps the answers in the score when a photograph is poor.
That is a robustness argument, not a performance one, and it is stated as such.

**At 15 slots** the share of cancer patients reaching the list is 0.65 for the questionnaire, 0.81 for the
image model and 0.82 for the fusion.

---

## 2 · The number that matters most

In PAD every cancer was biopsied, and the 566 development lesions that were *not* biopsied are all benign
— diagnosed by eye, and also the ones missing the history block of the questionnaire. They are easy cases
and they inflate every row above.

On the 116 validation patients whose lesions were **all** biopsied (93 of them with cancer):

| | Patient AUC |
|---|---|
| Final model | **0.890** |
| Image model | 0.886 |
| Questionnaire | 0.846 |

The nine-point gap becomes four, and it is no longer significant. **This is the number to quote when
asked how the system performs on hard cases.**

---

## 3 · M2 and the capacity ceiling

At a fixed number of slots, sensitivity is a joint statement about the model *and* the budget: no ranking
can exceed `budget ÷ prevalence`.

| Slots per 100 | Prevalence | Ceiling | Delivered |
|---|---|---|---|
| 5 | 10% | 0.50 | 0.35 |
| **10** | **10%** | **1.00** | **0.61** |
| 15 | 10% | 1.00 | 0.82 |
| 10 | 20% | 0.50 | 0.39 |

The charter's target of ≥ 0.90 is not reached at 10 slots. The constraint is capacity, not ranking:
**reaching 0.90 requires more urgent slots than the primary scenario allows**, and that number is reported
as a product requirement rather than recorded as a model failure.

---

## 4 · What the data can and cannot support

| | |
|---|---|
| ISIC clinical close-ups | 9,316 → **6,539** candidates after excluding PAD's ISIC copy (2,298) and the unlabelled benchmark (479) |
| ISIC after preprocessing | **6,404** images — minus 124 Indeterminate, 9 collision lesions and 2 duplicate photographs — 3,687 malignant (57.6%), 470 melanomas |
| PAD-UFES-20 | 2,298 images · 1,891 lesions · 1,373 patients · 52 melanomas · 47.4% malignant |
| Frozen PAD split | train 1,440 images / 878 patients · val 357 / 220 · **test 501 / 275** |
| Melanoma in the test split | **9 images from 7 patients** — one miss moves sensitivity by 11 points |
| Fitzpatrick V–VI in PAD | **11 images in total**, two of them in the test split |

**Dark skin cannot be evaluated on this data.** ISIC holds 376 V–VI clinical close-ups, but all come from
a single contributor and all sit in the training pool, so a gap measured there would be measuring the
institution as much as the skin tone. The slice is reported as raw counts and a dark-skin evaluation set
is the first requirement for any pilot.

---

## 5 · Leakage, shortcuts and the things that would have inflated the numbers

**Splitting.** Patient-level everywhere, frozen once and verified on every later run as assertions rather
than printed reports. A naive image-level split would put **29.3%** of PAD test images in training through
their own lesion, and 1.0% of ISIC.

**Duplicate photographs.** MD5 over the 6,539 original downloads found nothing. A perceptual hash over all
pairs found four groups covering 13 images; two of them are pairs with an *identical* hash, both
basal-cell carcinoma, each filed under two different `lesion_id` values — the same photograph stored twice
under different compression, which is exactly what a byte comparison cannot see. Both are excluded. The
other two groups carry contradictory diagnoses and are a property of the hash, not of the photographs.

**Contamination between the training pool and the target.** No ISIC training image is a copy of a PAD
photograph: five candidates at hash distance 2, every one rejected on inspection. Four of the five are the
same low-detail images that formed the degenerate hash groups above.

**The contributor shortcut.** Inside ISIC a classifier given *nothing but the contributor name* reaches
**AUC 0.729**. Any ISIC-internal score must be read against that number. Two consequences were taken:
model selection happens on PAD patients and never on the ISIC validation split, and the largest benign
contributor (99.4% benign, 1,241 images) is excluded from training entirely and used only for a
false-alarm check.

**Missingness that predicts the label.** In PAD `grew` is unanswered for 302 development lesions, 87.4% of
them malignant, and `changed` for 298, 88.3% malignant — administrative, not clinical. A model given
nothing but the count of missing optional fields reaches **AUC 0.839**. Keeping missingness as a feature
category lifts the questionnaire from 0.860 to 0.916, and **that 0.056 is the size of the artefact**. No
missingness indicator reaches any model; answers are imputed inside the pipeline, fitted per fold.

---

## 6 · The questionnaire

Cross-validated on development patients, patient-grouped:

| Variant | AUC | Coverage |
|---|---|---|
| A · missingness as a category | 0.916 ± 0.018 | 100% — contains the artefact |
| **D · all lesions, imputed, no indicator** | **0.860 ± 0.021** | 100% — the policy the project uses |
| B · complete cases only | 0.893 ± 0.029 | 80%, on a population that is 32.8% malignant rather than 44.0% |
| C · count of missing optional fields alone | 0.839 ± 0.012 | the artefact on its own |

On the frozen validation split, variant D reaches **patient AUC 0.867**. A gradient boosting variant
reaches 0.890 on the same patients, but it was *selected* on them, so it is reported in notebook 04 as a
study and the logistic regression remains the bar.

**Self-reported answers cost real sensitivity.** In PAD the answers were recorded by senior medical
students during a visit; in the app the patient answers alone. Flipping 10% of the answers at random
lowers patient AUC from 0.867 to **0.823** and primary-scenario sensitivity from 0.52 to 0.46;
over-reporting only ("no" → "yes") gives 0.835 and 0.49. **Every questionnaire number here is an upper
bound for the app.**

**What the model learned**, as odds ratios, yes versus no: bleeding 6.37 · elevation 6.00 · changed 3.81 ·
hurt 2.53 · itch 1.34 · age 1.28 per ten years. `grew` comes out at 0.89, below one — an artefact of
imputing "no" into a mostly malignant group, not a clinical finding, and reported as such.

---

## 7 · The image model

ConvNeXt-Tiny at 448 px: a head warm-up and full training on **4,325 ISIC clinical close-ups**, then
fine-tuning on **1,430 photographs of the 878 PAD training patients**. Six classes — MEL, SCC, BCC, AK,
NEV, BKL — collapsed to one cancer score, P(MEL)+P(SCC)+P(BCC), the charter's M2 definition.

**Most of the clinic-to-phone gap was actinic keratosis.** The ISIC-only model reaches 0.778 overall but
**0.899** once the 81 patients who have an actinic keratosis are removed. ISIC holds 123 AK photographs
and PAD holds 730: the model trained only on clinic images calls pre-cancer cancer, and fine-tuning on
phone photos is largely learning that boundary.

**Decision rules were fixed before the runs, and both outcomes are reported.** Doubling the SCC class
weight was adopted on its stated criterion (SCC photos called AK fell from 7.3 to 5.7, SCC recall rose
from 0.41 to 0.52, AUC fell 0.953 → 0.950, within the 0.01 allowance). Averaging three seeds was
**rejected**: the paired gain was +0.0006, CI −0.007 to +0.008.

**Where it is still wrong.** Eight of 34 SCC photographs are called actinic keratosis, which the charter
counts as non-cancer. AK can progress to SCC, so this is the hardest and most consequential boundary the
system draws.

**A confound that inflates the patient-level numbers.** Cancer patients in PAD have more photographs than
benign patients (1.93 against 1.33 on average), and the photo count alone reaches AUC 0.663. Under the
maximum rule, more photographs mean more chances.

---

## 8 · The domain gap, measured at the training resolution

250 images per group, all put through the identical preprocessing:

| | ISIC · MILK | ISIC · MSK | PAD · phone |
|---|---|---|---|
| Sharpness (Laplacian variance) | 213.3 | 361.6 | **83.1** |
| Saturation | 108.9 | **55.5** | 78.0 |
| Contrast | 18.5 | 23.5 | 22.8 |
| Brightness | 136.9 | 138.3 | 145.1 |
| Long side below 512 px | 0.0% | 0.0% | **25.2%** |

Phone photographs are 2.6× less sharp than MILK and 4.4× less sharp than MSK, while brightness and
contrast barely move: **that is blur, not exposure.** The phone photos are also *higher* resolution than
most of the training pool, so the blur is real and not an artefact of resampling.

The two ISIC contributors differ from each other as much as either differs from PAD. MSK carries half of
MILK's colour saturation and is 99.4% benign, so colour alone is a usable shortcut to the label — the
measured form of the contributor confounding above. Blur and that colour fingerprint are the two things
augmentation has to attack.

---

## 9 · Not yet measured

- **The held-out test set.** One run, after the team approves `reports/decisions.md`.
- **Surgical-marker ink**, a known shortcut visible on some PAD photographs. An automatic detector failed
  validation. After the test run the test photographs are labelled for ink *blind* to diagnosis and score,
  and benign inked photographs are compared with benign photographs without ink.
- **Groundedness (M3)** of the agent's guidance, on the frozen set of 20 questions including 5 that the
  corpus cannot answer.
- **Running cost** at the assumed volume of 100 arrivals a day.
