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
| **M1** | Median simulated days to appointment for patients with a malignant lesion, under the 10% waiting-list prioritization policy | ≤ 7 days against the arrival-order baseline — **met**: 24.0 → 1.0 days in the primary scenario. The median hides a split distribution; see the note below (`src/simulate_queue.py`, `reports/m1_queue_simulation.json`) |
| **M2** | Patient-level sensitivity for skin cancer (MEL + BCC + SCC) among patients flagged for priority review, evaluated on held-out PAD-UFES-20 patients | ≥ 0.90 (not met; held-out test sensitivity: 0.664 in the primary evaluation scenario) |
| **M3** | Groundedness of the agent's guidance on a frozen set of 20 questions, including 5 unanswerable questions | ≥ 0.90 groundedness and 5/5 refusals. **Groundedness target met:** 0.933 after re-ranking. The 5/5 refusal result was measured before re-ranking only; refusal behavior after re-ranking has not been independently re-tested. The groundedness metric verifies citation IDs against retrieved passages, not full semantic accuracy. (`notebooks/07_rag_nci_m3.ipynb`) |
| **M4** | Agent decisions on a frozen set of 15 complex workflow cases, measured against an if-else script on the same cases | ≥ 13/15 correct and above the baseline, with no patient suppressed, downgraded or removed. **Not met on accuracy**: agent 9/15, system after guardrails 11/15, if-else 12/15. **Met on safety**: no patient removed, no score or rank changed, no reassuring wording (`notebooks/08_agent_m4_vs_ifelse.ipynb`) |


**M1 met its simulated median criterion; M2 and M4 did not meet their targets. M3 met its measured groundedness target, while refusal behavior after re-ranking remains to be independently verified.** M2 is a
capacity statement as much as a ranking one — see the note below. M4 is the more interesting failure: the
agent does not beat a rule script on decisions, which is a finding about where an agent belongs rather than
a bug to be tuned away. M1 is met on the median, but the median is the weakest part of that metric and the
note below says why. `reports/results.md`, `reports/m1_queue_simulation.json` and
`notebooks/08_agent_m4_vs_ifelse.ipynb` carry the diagnoses.

**Prioritization policy.** Every morning, the system ranks all patients currently on the dermatologist's waiting list by their combined risk score and flags the highest-risk **10% of the waiting list** for priority review. The number of flagged patients scales with the queue size (e.g., 5 out of 50, 10 out of 100, or 30 out of 300). The policy applies to the entire waiting list at the time of ranking, not only to new daily arrivals. Patients who are not flagged retain their existing appointments. The dermatologist makes all clinical and scheduling decisions.

**M1, and what the median hides.** A simulated clinic: 100 appointments a day, 100 arrivals, 10%
prevalence, the waiting list re-ranked every morning by the final model, and 10 of the day's appointments
kept for priority review. Patients are resampled from the 220 validation patients carrying their real model
scores. Test scores were not re-opened for this, and the charter already frames queue metrics as scenario
results rather than predictions of fund performance. Against the arrival-order baseline of 24.5 days, the
median wait for a patient with a malignant lesion falls to **1.0 day**, and M1 is met.

The median is the wrong summary on its own, because the distribution has no middle. A patient either
reaches an urgent slot and is seen the next morning, or does not and waits the full queue. At 10 urgent
slots **61%** reach one — the same figure M2 reports as sensitivity, arrived at here independently, by
simulating a queue day by day. That is what puts the median in the fast group. At 5 slots only 41% reach a
slot, the median returns to **25.0 days**, and the policy becomes slightly **worse than doing nothing**:
the urgent slots come out of the routine queue that everyone else, including the other 59%, is waiting in.

Three numbers belong beside the median. **44%** of malignant patients still wait more than 7 days. The 90th
percentile is **27 days**, against 24 at baseline. And the median wait for a benign lesion rises from 24 to
**27 days**. The policy moves cancer patients forward by moving everyone else back, and at 30 urgent slots —
where only 5% of cancer patients are left over target — a benign patient waits 34 days. The real question
is therefore not how many slots the model needs, but how much longer the clinic is willing to make everyone
else wait. That is a product decision, and the simulation is there to price it.

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
| **NCI PDQ** Health Professional summaries | Guidance corpus for the Agent (RAG) | 7 documents · 224 sections · 724 chunks | NCI public-domain content; credit NCI |


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
                                      Agent: evidence, citations and
                                          one of five decisions
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

`agent/` holds the whole layer — tools, prompts, decision loop and guardrails. Notebooks 07, 08 and 09
build it, test it and run it, and every figure below comes from their saved output in `reports/agent/`.

The build is a CrewAI crew, *Clinician Seat triage evidence assistant*. It takes one patient at a time from
the list the model ranked, assembles the evidence a dermatologist needs before the appointment, cites the
guidance it used, and returns one of five decisions through a structured JSON contract. The same role, goal
and task text live in CrewAI Studio, where the prompt was designed; the code adds what Studio cannot do.

**Which tools it may call.** Seven, registered in `build_crew()`, and nothing else:

| Tool | Returns |
|---|---|
| `get_case_data` | the locked rank, combined risk score and band, age and body site — read-only |
| `predict_visual_risk` | the image model's risk band for this photo, already computed |
| `score_symptoms` | the six symptom answers, the symptom model's band, and the patient's own note (data, not instructions) |
| `check_image_quality` | the photo's status — `usable`, `borderline`, `unusable`, `not_a_lesion` — and the specific issues |
| `retrieve_guidance_context` | up to three passages from the frozen NCI PDQ corpus, each with a chunk id, for citation |
| `request_retake` | queues a Hebrew retake instruction as a **draft** |
| `ask_patient` | queues one Hebrew follow-up question as a **draft** |

The last two only write to an outbox. Plain code decides whether a draft is ever sent, and only for the
decisions `request_retake` and `ask_patient`, after every check has passed. `draft_biopsy_referral` is
below the charter's cut line and is not implemented.

The retrieval query is built **by code** from the symptoms answered yes and the body site, not by the model:
retrieval stays reproducible, and a photo word such as "too dark" cannot pull passages about dark skin.
Patients 18 and over are never served the two childhood pages.

**What happens when it fails.** `review_case` retries once on malformed output, never retries a call that
has hung, and after 120 seconds falls back to `manual_review` with the fixed model-evidence summary and the
patient's priority unchanged. Every outcome, attempt count and duration goes to the agent log.


**Validation Agent run — 31 October 2026.**
The final committed run in `notebooks/09_agent_daily_report.ipynb` processed all 10 selected PAD validation patients successfully, with **10/10 Agent outcomes, zero retries and zero fallbacks**.

The Agent produced 8 `proceed` and 2 `proceed_with_flag` decisions. No guardrail corrections were required, and patient scores and ranking positions remained unchanged.

The median execution time was **6.85 seconds per patient**. The longest execution was **72.7 seconds (PAT_393)**, within the configured 120-second timeout.

The report and matching execution log are committed in `reports/agent/`.

Earlier, separate runs demonstrated the failure path: recovery from an API quota error, and complete report generation through the fallback mechanism when the Agent was unavailable. These earlier results are historical evidence and are not part of the final validation run.


**The system works without it.** `USE_AGENT = False`, and `agent/triage_agent.py --no-agent`, produce the
same report from the same ranking, with a fixed per-patient summary and the decision `model_evidence_only`.
Both the agent and the non-agent report are in `reports/agent/`, for the sample list and for the validation
list.

**The agent cannot change a score or a rank** — not as an instruction, but because the `CaseReview` schema
it answers in has no score field and no rank field. There is nothing for it to change. The model ranks; the
agent explains and prepares the next step.

**Guardrails are code, not prompt.** `enforce_guardrails` runs on every answer: reassuring words are
rejected in English and Hebrew, with Hebrew final letters normalised and NCI quotes excluded so a clinical
passage is not mistaken for reassurance; a patient under 18, a photo that may not show a lesion, or a
missing critical field each force `manual_review`; a citation survives only if its chunk id was retrieved
for that patient **and** its quote appears word for word in that chunk; every number in the summary must
come from the case record or a surviving citation; and a message to a patient leaves the system only for
`request_retake` and `ask_patient`, after all the checks have passed.

**What the agent decides that a fixed script could not — the honest answer.** On decisions alone, nothing.
On 15 frozen scenarios (M4) an if-else script on the same fields scored 12/15, the agent 9/15, and the agent
behind the code guardrails 11/15; its decisions also vary between runs of the same case. Where it did add
judgement was the one case the rules get wrong for a reason rules cannot encode: a lesion behind the ear,
where a retake would not help, so it proceeded with a flag instead of demanding one. What it adds reliably
is the patient-specific text — the question that targets the actual gap in the form, the retake instruction
in Hebrew, the grounded citation. The conclusion this project draws is to let code make every rule-shaped
decision and leave the agent the texts and the cases the rules do not cover.

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
- **RAG evaluation (M3).** A frozen set of 20 questions, including 5 unanswerable questions, was used to evaluate citation grounding and refusal behavior. Citation-ID groundedness improved from 0.733 to 0.933 after re-ranking. All 5 refusal tests passed before re-ranking; the refusal test has not yet been repeated after re-ranking. Citation-ID matching does not independently establish semantic or clinical correctness.


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
│  ├─ 05_image_model.ipynb   the image model and the fusion
│  ├─ 06_test_run.ipynb      the single held-out test run, 30.10.2026
│  ├─ 07_rag_nci_m3.ipynb    the NCI PDQ corpus, retrieval and groundedness (M3)
│  ├─ 08_agent_m4_vs_ifelse.ipynb   the agent against the if-else baseline (M4)
│  └─ 09_agent_daily_report.ipynb   the daily report for the dermatologist
├─ src/                      the runnable model pipeline (see src/README.md)
│  └─ build_daily_cases_from_val.py   validation-only model-to-agent demo
├─ agent/                    tools, prompts, the loop, guardrails
├─ tests/                    regression tests for the shared band policy         
├─ evaluation/               eval sets, groundedness, error analysis
├─ demo/                     the three-case walkthrough
├─ models/                   model artefacts
│  ├─ figures/
│  ├─ decisions.md           the frozen decisions for the single test run
│  ├─ results.md             the baseline table and the key results
│  └─ results_image_model.md the image model's own report
└─ reports/                                                      
```

## 9 · Running it

```bash
pip install -r requirements.txt
```

The notebooks run in Google Colab and mount Drive; each one states what it needs and fails with a message
naming the notebook that produces it. **Run order: 01 → 02 → 03 → 04 → 05.** Notebooks 01 and 03 both
fetch the PAD photographs from the same mirror, so whichever runs first does the work.
Notebooks 06 to 09
run after those five: 06 is the single held-out test run, 07 builds the guidance corpus and measures M3, 08
compares the agent with the if-else baseline, and 09 produces the daily report. 07 and 09 read a
`GEMINI_API_KEY` from Colab Secrets; neither needs the image archive, although the photo-review cell in 09
reports the files as missing without it.

`src/` holds the runnable image-model pipeline, documented in [src/README.md](src/README.md). The evaluation scripts require a CUDA GPU and the image data in the configured external data directory.

The decisions governing the single held-out test run were frozen in `reports/decisions.md`. The test was executed once on 30.10.2026 in `notebooks/06_test_run.ipynb`, which keeps the run's own output, and its results are preserved in `reports/test_results.json` and summarized in `reports/results.md`.

### Validation-only model to agent demonstration


The reproducible bridge is `src/build_daily_cases_from_val.py`, executed as a Python module from the repository root.
It uses **stored validation predictions** by default, not the held-out test set.
It restricts the 220 validation patients (103 malignant; 46.8%) to 138 with
exactly one lesion and one photo (47 malignant; 34.1%), so the image, symptom
data and displayed photo refer to the same lesion. A seeded simulation draws
100 of those eligible patients with 10 malignant cases (10% scenario
prevalence), ranks all 100 using the frozen **60% image + 40% questionnaire**
percentile-rank fusion and exports the top 10. Controlled prevalence does
**not** correct for selection bias: this subset is for technical integration,
not an estimate of clinical performance or generalization.

Run from repository root (also works in Colab with `os.chdir(PROJECT)`):

```bash
python -m src.build_daily_cases_from_val
python agent/triage_agent.py --cases data/processed/daily_cases_val.csv --no-agent --out reports/agent/validation_smoke.docx
```

The second command requires `python-docx` (install with `pip install
python-docx`). It is a **smoke test without an LLM agent**. The output
`data/processed/daily_cases_val.csv` has ranked patients and model evidence;
the audit `data/processed/val_100_candidates_audit.csv` contains diagnoses
for evaluation by the project team only and must **never** be supplied to
the agent. The exported relative `risk_score`/`risk_band` are **not calibrated
malignancy probabilities or validated clinical risk categories**.


**Image-quality annotations for the validation demonstration.**

The ten selected images were annotated through an AI-assisted visual review of screenshots. These annotations are used for technical demonstration only. They were not produced by a validated image-quality classifier or independently reviewed by a dermatologist.
The annotations are stored in `data/processed/val_top10_quality_template.csv`.
For future runs, original photographs should be reviewed and quality labels validated before they are used by the Agent.
The following image-quality categories are supported:


| `image_quality` | Labeling rule |
| --- | --- |
| `usable` | The lesion is visible, in focus, and adequately illuminated. |
| `borderline` | The lesion is visible, but a photo issue may limit assessment. |
| `unusable` | The photo does not support reasonable visual assessment. |
| `not_a_lesion` | No skin lesion relevant to the task can be identified. |

Use `image_issues` for specific issues (semicolon-separated), with the
recognizable terms **`blurred`, `too dark`, `too bright`, `too far`, `glare`,
`not centred`**; use an empty cell if no issue. Do not substitute `blurry`
for `blurred`: the retake rules recognize the latter spelling.


The committed validation annotations contain 8 `usable` images and 2 `borderline` images with `glare`.
To reproduce the daily Agent input using these saved annotations, run the following command from the repository root:


```bash
python -m src.build_daily_cases_from_val --quality-file data/processed/val_top10_quality_template.csv
```

In `notebooks/09_agent_daily_report.ipynb`, set `CASES_FILE` to
`os.path.join(PROCESSED, "daily_cases_val.csv")` and `USE_AGENT = True`
only after the photo reviews; confirm the per-patient agent log rather than
assuming a generated Word file proves the agent ran. With image files, weights
and dependencies installed, `--fresh-inference` can instead run the trained
models on the PAD validation photos; it creates separate `_fresh` outputs
and a fresh quality template. **Do not upload the original photographs**
as part of this demo or treat the report as a clinical recommendation.


The builder in `src/` and the legacy `agent/make_daily_list.py` helper use the same queue-relative risk bands defined in `src/fusion.py`. The older helper recomputes fusion within its sampled day and defaults unchecked photo quality
to `not_checked`, rather than silently treating it as usable.

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
