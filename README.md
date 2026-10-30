# Skin Lesion Triage for Healthcare Providers

**A dermatologist's waiting list, re-ordered by risk.** A patient photographs a skin lesion with their phone
inside the health-fund app and answers six questions. Every morning, the system ranks all patients currently
on the dermatologist's waiting list and flags the highest-risk 10% for priority review. The dermatologist
reviews these cases and decides whether an earlier appointment or further assessment is warranted.
The system does not make clinical decisions autonomously.

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
| **M1** | Median simulated days to appointment for patients with a malignant lesion, under the 10% waiting-list prioritization policy | ≤ 7 days in the primary scenario, reported against the arrival-order baseline (not yet evaluated) |
| **M2** | Patient-level sensitivity for skin cancer (MEL + BCC + SCC) among patients flagged for priority review, evaluated on held-out PAD-UFES-20 patients | ≥ 0.90 (not met; held-out test sensitivity: 0.664 in the primary evaluation scenario) |
| **M3** | Groundedness of the agent's guidance on a frozen set of 20 questions, 5 of them unanswerable | ≥ 0.9, and 5/5 refusals — **met**: 0.933 after re-ranking, 5/5 refusals, and every number in an answer traced to its cited chunk (`notebooks/07_rag_nci_m3.ipynb`) |
| **M4** | Agent decisions on a frozen set of 15 complex workflow cases, measured against an if-else script on the same cases | ≥ 13/15 correct and above the baseline, with no patient suppressed, downgraded or removed. **Not met on accuracy**: agent 9/15, system after guardrails 11/15, if-else 12/15. **Met on safety**: no patient removed, no score or rank changed, no reassuring wording (`notebooks/08_agent_m4_vs_ifelse.ipynb`) |

Two of the four are met, and the two that are not are reported with the number they reached. M2 is a
capacity statement as much as a ranking one — see the note below. M4 is the more interesting failure: the
agent does not beat a rule script on decisions, which is a finding about where an agent belongs, not a bug
to be tuned away. `reports/results.md` and `notebooks/08_agent_m4_vs_ifelse.ipynb` carry the diagnosis.

**Prioritization policy.** Every morning, the system ranks all patients currently on the dermatologist's waiting list by their combined risk score and flags the highest-risk **10% of the waiting list** for priority review. The number of flagged patients scales with the queue size (e.g., 5 out of 50, 10 out of 100, or 30 out of 300). The policy applies to the entire waiting list at the time of ranking, not only to new daily arrivals. Patients who are not flagged retain their existing appointments. The dermatologist makes all clinical and scheduling decisions.

**M2 and the capacity ceiling.** The primary evaluation scenario assumes 10 priority-review slots per 100 arrivals and a 10% malignancy prevalence. This is a standardized, prevalence-weighted evaluation scenario representing 10% review capacity, rather than a full simulation of a changing waiting list.
At this operating point, the theoretical sensitivity ceiling is 1.00. The final model achieved **0.664 sensitivity (95% CI: 0.570–0.750)** on held-out test patients, compared with 0.612 on validation. The pre-specified M2 target of ≥ 0.90 was therefore **not met**.
Sensitivity depends on both ranking performance and prioritization capacity. Since the theoretical ceiling at the primary operating point is 1.00, the gap cannot be attributed to capacity alone.
Increasing capacity to 15 slots per 100 arrivals raised test sensitivity to 0.820. On validation, sensitivity reached 0.903 at 20 slots per 100 arrivals; this latter result was not confirmed on the held-out test set.
The product's prioritization policy remains unchanged at **10% of the current waiting list**. Increasing this percentage would require a separate product decision and additional evaluation.

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
Phone photo ──→ Image model ───────────────┐
                                          ├──→ Rank fusion (60/40)
Symptom form ─→ Questionnaire model ───────┘             │
                                                        ▼
                                           Top 10% of waiting list
                                                        │
                                                        ▼
                                          Agent (planned): evidence
                                               and citations
                                                        │
                                                        ▼
                                         Dermatologist: clinical review
```

Scores are aggregated **image → lesion → patient by maximum**, frozen in
`data/processed/aggregation_policy.json`: a patient's priority is set by their most suspicious lesion, and
averaging would dilute it.

## 5 · The baseline table — "the cheapest thing that works"

**The held-out result first.** The final evaluation was performed once, on 30.10.2026, using 275 held-out PAD patients, 128 of whom were labelled as having a malignant lesion. The evaluation followed the decisions frozen in `reports/decisions.md`.
All models were compared using the same patient-level aggregation rule and the same primary evaluation scenario: 10 priority-review slots per 100 arrivals at an assumed 10% malignancy prevalence. This standardized scenario evaluates ranking performance under a defined review capacity; it does not simulate the complete dynamics of the real waiting list.

| Model | Patient AUC (95% CI) | Sensitivity | PPV |
|---|---|---|---|
| Arrival order — today's practice | 0.500 | 0.10 | 0.10 |
| **Questionnaire only — the baseline** | **0.862** (0.813–0.902) | **0.57** | **0.57** |
| ConvNeXt + PAD fine-tuning | 0.953 (0.929–0.972) | 0.69 | 0.69 |
| **Final: image 60% + questionnaire 40%** | **0.954** (0.928–0.975) | **0.66** | **0.68** |

Paired on these patients: image − questionnaire **+0.092** (+0.052 to +0.136); fused − image **+0.000**
(−0.014 to +0.015). Both reproduce the validation figures almost exactly — including the second, which was
written down in advance as the one expected to be zero.

**And the table that chose the model.** All rows below are on the 220 PAD validation patients (103
malignant), at the same operating point. DINOv2 and the ISIC-only model were never run on the test
patients: they are here to show what the expensive model had to beat.

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
and significant. **On the held-out patients it held: +0.092, and 69% against 57%.**

**And one thing it hides.** In PAD every cancer was biopsied, and the lesions that were not biopsied are
all benign — easy cases that inflate every row. On the **138 test patients** whose lesions were all
biopsied (115 with cancer) the AUCs are **0.859** (image), **0.851** (final) and **0.767**
(questionnaire); on the 116 validation patients they were 0.886, 0.890 and 0.846.

On validation the nine-point lead over the questionnaire shrank to four and lost significance, and this
README said so before the test was opened. On held-out data it did not shrink: **+0.093, interval +0.005
to +0.178** — clearing zero by 0.005, on 23 benign patients. That is a real result and a fragile one, and
it belongs in print with its lower bound rather than as "significant". At the operating point the
difference is plainer: on these patients the questionnaire puts **22%** of cancer patients on the daily
list and the image model puts **46%**.

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
├─ src/                      the runnable image-model pipeline (see src/README.md)
├─ agents/                   tools, prompts, the loop, guardrails          
├─ evaluation/               eval sets, groundedness, error analysis
├─ models/                   model artefacts
├─ reports/
│  ├─ figures/
│  ├─ decisions.md           the frozen decisions for the single test run
│  ├─ results.md             the baseline table and the key results
│  └─ results_image_model.md the image model's own report
└─ presentation/                                                           
```

## 9 · Running it

```bash
pip install -r requirements.txt
```

The notebooks run in Google Colab and mount Drive; each one states what it needs and fails with a message
naming the notebook that produces it. **Run order: 01 → 02 → 03 → 04 → 05.** Notebooks 01 and 03 both
fetch the PAD photographs from the same mirror, so whichever runs first does the work.

`src/` holds the runnable image-model pipeline, documented in [src/README.md](src/README.md). The evaluation scripts require a CUDA GPU and the image data in the configured external data directory.

The decisions governing the single held-out test run were frozen in `reports/decisions.md`. The test was executed once on 30.10.2026 in `notebooks/06_test_run.ipynb`, which keeps the run's own output, and its results are preserved in `reports/test_results.json` and summarized in `reports/results.md`.

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
