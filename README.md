# Skin Lesion Triage for Healthcare Providers

**A dermatologist's waiting list, re-ordered by risk.** A patient photographs a mole with their phone
inside the health-fund app and answers six questions. Every morning the dermatologist opens a list of up
to ten patients whose lesions look most suspicious, moves them to an earlier appointment, and — where the
evidence supports it — issues a biopsy referral before the visit.

**BIU DS23 capstone.** Ifat Davidson (Product) · Yuval Rubenchuk (Data) · Roi Budnitsky (Model) ·
Tami Khazma (Agent)

---

## 1 · The problem, and who has it

A dermatologist in a health fund sees patients in arrival order. A melanoma that waits eleven weeks is a
melanoma that has grown for eleven weeks, and nothing in today's queue distinguishes it from a benign
mole booked the same morning. The cost is measured in weeks of delay for the patients who can least
afford them.

The system does not diagnose and does not replace the appointment. It changes **who is seen first**, and
it hands the dermatologist the evidence for that ordering.

## 2 · Success criteria

| | Metric | Target |
|---|---|---|
| **M1** | Median simulated days to appointment for patients with a malignant lesion, under the fixed urgent-slot capacity | ≤ 7 days in the primary scenario, always reported against the arrival-order baseline |
| **M2** | Patient-level sensitivity for skin cancer (MEL + BCC + SCC) among the patients the dashboard flags, on held-out PAD-UFES-20 patients | See the capacity note below |
| **M3** | Groundedness of the agent's guidance on a frozen set of 20 questions, 5 of them unanswerable | ≥ 0.9, and 5/5 refusals |

**M2 and the capacity ceiling.** At a fixed number of urgent slots, sensitivity is a joint statement about
the model *and* the slot budget: no ranking can exceed `budget ÷ prevalence`. At 10 slots per 100 arrivals
and 10% prevalence the ceiling is 1.00. The delivered model reaches **0.60** there and **0.81** at 15 slots
on validation patients. The number of slots required to reach 0.90 is therefore reported as a product
requirement rather than treated as a model failure.

## 3 · Data

| Source | Role | Size | Licence |
|---|---|---|---|
| **ISIC Archive**, clinical close-ups incl. MILK10k | image-model training | 9,316 clinical close-ups → **6,539** candidates after removing PAD and the unlabelled benchmark → **6,404** after preprocessing exclusions | CC-BY-NC / CC-BY / CC-0 per image |
| **PAD-UFES-20** | target domain: real phone photos, and the only source with the symptom answers the app collects | 2,298 images · 1,891 lesions · 1,373 patients · 52 melanomas · 47.4% malignant | CC BY 4.0 |
| **NCI PDQ** patient summaries | guidance corpus for the agent | ~10–20 documents | free of copyright, credit NCI |

Raw images are **not** in this repository. `notebooks/01_eda_isic.ipynb` downloads ISIC through the
official CLI and PAD from a public mirror, both verified on content; `data/data_dictionary.md` describes
every table and column that the pipeline writes.

**The de-duplication rule, frozen.** PAD appears twice: on Mendeley and inside ISIC as collection 406. The
whole ISIC copy is excluded from training, because in ISIC those images carry no PAD patient id — training
on them would put photographs of held-out test patients into the training set with no way to detect it.
The exclusion is implemented on contributor attribution and a missing `diagnosis_1`, and both proxies were
verified against ISIC collections 406 and 424 through the API: **identical id sets, zero differences in
either direction.**

## 4 · What the pipeline does

```
phone photo ─┐
             ├─→ image model (ConvNeXt-Tiny, 448 px) ──┐
symptom form ┘                                         ├─→ rank fusion (60/40) ─→ daily top-10
             └─→ questionnaire model (logistic reg.) ──┘                              │
                                                                                      ▼
                                                        agent: evidence, citations, guardrails
```

Scores are aggregated **image → lesion → patient by maximum**, frozen in
`data/processed/aggregation_policy.json`: a patient's priority is set by their most suspicious lesion, and
averaging would dilute it.

## 5 · The baseline table — "the cheapest thing that works"

All rows on the same **220 PAD validation patients** (103 malignant), the same aggregation rule, the same
operating point: 10 urgent slots per 100 arrivals at 10% prevalence. Test patients have not been opened.

| Model | Patient AUC (95% CI) | Sensitivity | PPV | Cost to build |
|---|---|---|---|---|
| Arrival order — today's practice | 0.500 | 0.10 | 0.10 | none |
| **Questionnaire only — the baseline** | **0.867** (0.815–0.913) | **0.52** | **0.53** | seconds to fit, no photo needed |
| DINOv2 frozen + logistic regression | 0.880 (0.835–0.923) | 0.49 | 0.51 | no training, one forward pass |
| ConvNeXt, ISIC only | 0.778 (0.719–0.836) | 0.30 | 0.30 | ~1 GPU-hour |
| ConvNeXt + PAD fine-tuning | 0.956 (0.930–0.977) | 0.60 | 0.61 | ~2 GPU-hours |
| **Final: image 60% + questionnaire 40%** | **0.954** (0.928–0.978) | **0.61** | **0.61** | the above, fusion is unfitted |

Paired differences: image − questionnaire **+0.089** (+0.047 to +0.132); fused − image −0.001
(−0.014 to +0.011).

**Three things this table is meant to show.**

The questionnaire alone already reaches 0.867 and needs no photograph at all — it is the bar, and it was
specified before any model was fitted and never tuned on validation. A frozen DINOv2 with a logistic
regression on top matches it without training a single weight. The expensive model earns its place by
+0.089 AUC and by putting 60% of cancer patients on the daily list instead of 52%, and that gain is paired
and significant.

**And one thing it hides.** In PAD every cancer was biopsied, and the 566 development lesions that were
not biopsied are all benign — easy cases that inflate every row. On the 116 validation patients whose
lesions were all biopsied (93 with cancer) the AUCs are **0.890** (final), **0.886** (image) and
**0.846** (questionnaire). The honest gap on hard cases is four points, not nine.

## 6 · The agent

> *To be completed by the Agent hat before the repo freeze.* `agents/` must contain the tool definitions,
> the prompts, the decision loop and the guardrails, plus a paragraph answering the three questions the
> charter requires: what the agent decides that a fixed script could not, which tools it may call, and what
> happens when it fails. The system must work without it.

The current build is a CrewAI crew, *Clinician Seat triage evidence assistant*: for one patient on today's
urgent list it assembles the evidence a dermatologist needs before the appointment, cites the guidance it
used, and returns a single decision with a structured JSON contract.

## 7 · Evaluation

`evaluation/` holds the frozen eval sets and the error analysis.

- **Leakage.** Patient-level splitting everywhere, frozen manifests created once and verified on every
  later run, and the checks are assertions rather than printed reports. A naive image-level split would
  leak 29.3% of PAD test images and 1.0% of ISIC. Near-duplicate photographs were hunted with MD5 **and**
  a perceptual hash: the byte comparison found nothing, the perceptual hash found two pairs filed under
  two lesion ids each, and they are excluded. No ISIC training image is a copy of a PAD photograph —
  five candidates at hash distance 2, all rejected on inspection.
- **The contributor shortcut.** Inside ISIC a classifier given nothing but the contributor name reaches
  **AUC 0.729**. Any ISIC-internal score must be read against that number, which is why model selection
  happens on PAD patients and why the largest benign contributor is excluded from training entirely.
- **Self-reported answers.** The PAD questionnaire was filled in by medical students during a visit; app
  users answer alone. Flipping 10% of the answers lowers patient AUC from 0.867 to 0.823. Every
  questionnaire number here is therefore an upper bound.
- **Groundedness (M3).** Twenty frozen questions, five of them unanswerable from the corpus, with every
  cited number checked against its source.

## 8 · Repository

```
├─ README.md                 this file
├─ CHARTER.md                the plan, plus the checkpoint updates
├─ DATA_DECISIONS.md         every dataset considered, and why it was kept or rejected
├─ requirements.txt
├─ data/
│  └─ data_dictionary.md     every table and column the pipeline writes
├─ notebooks/
│  ├─ 01_eda_isic.ipynb      ISIC acquisition, audit, duplicates, domain gap
│  ├─ 02_eda_pad.ipynb       PAD metadata, the frozen patient split, the questionnaire baseline
│  ├─ 03_preprocessing.ipynb frozen manifests and the modelling hand-off
│  ├─ 04_symptom_model.ipynb the questionnaire model
│  └─ 05_image_model.ipynb   the image model and the fusion
├─ src/                      reusable code for the image model
├─ agents/                   tools, prompts, the loop, guardrails
├─ evaluation/               eval sets, groundedness, error analysis
├─ models/                   model artefacts
├─ reports/
│  ├─ figures/
│  └─ results.md             the baseline table and the key results
└─ presentation/
```

## 9 · Running it

```bash
pip install -r requirements.txt
```

The notebooks run in Google Colab and mount Drive; each one states what it needs and fails with a message
naming the notebook that produces it. **Run order: 01 → 02 → 03 → 04 → 05.** Notebooks 01 and 03 both
fetch the PAD photographs from the same mirror, so whichever runs first does the work.

Every notebook asserts the figures this README quotes. If an assertion fires, the archive has changed and
the number is corrected here, not in the notebook.

**No secrets in this repository.** API keys are read from environment variables or Colab Secrets.

## 10 · Running cost, order of magnitude

> *To be completed before the final submission: monthly API, storage and compute for the agent layer and
> the inference path, at the assumed volume of 100 arrivals a day.*

## 11 · What we would not claim

- **Dark skin cannot be evaluated here.** PAD holds 11 Fitzpatrick V–VI images in total, two of them in
  the test split. ISIC holds 376, all from a single contributor and all in the training pool, so a gap
  there would measure the institution as much as the skin tone. The slice is reported as raw counts and
  a dark-skin evaluation set is the first requirement for any pilot.
- **Melanoma statistics are thin.** The frozen test split holds 9 melanomas; one miss moves sensitivity by
  11 points. M2 is stated for skin cancer overall and melanoma is reported separately.
- **Pre-cancer is the hardest boundary.** Actinic keratosis is scored as a negative, and the image model
  still calls some squamous cell carcinoma actinic keratosis. It is the most consequential error the
  system makes.
- **One programme, trained photographers.** PAD was shot by a clinical programme. Patients photographing
  themselves will produce worse photographs than anything measured here.
