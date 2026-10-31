# Project Charter · Skin Lesion Triage for Healthcare Providers

**Team:** Ifat Davidson · Tami Khazma · Roi Budnitsky · Yuval Rubenchuk | **Course:** BIU DS23 Capstone | **Written:** 04.10.2026
**Retrieval asset:** the module 5 grounded assistant.

> **How to read this document.** Sections 1–7 are the plan as it was registered on 4.10, before any model
> was fitted. Targets written there are left as they were written, including the two that were not met.
> Where a figure has since been superseded or a decision reversed, the change is marked inline as
> **[amended DD.MM]** with the reason, rather than edited away. Section 8 holds the checkpoint updates.
>
> **Amendments since 4.10, in one place:** the product policy and the evaluation scenarios are now stated
> separately and both are pinned to numbers (§1, §2); HAM10000 was dropped (§3); the metadata-floor figures
> were superseded when the missingness policy was frozen (§4); the agent's decision is restated as
> escalation-only (§5); two risks were added (§6). The safety principle in §1 did not survive the M1
> simulation unchanged — see the note there.

## 1 · Problem & User
**User:** A dermatologist working within an Israeli healthcare fund (e.g., Clalit, Maccabi) who manages an overloaded appointment queue.

**Pain:**  the dermatologist cannot tell which patient on the waiting list cannot wait.
A public-system dermatologist appointment in Israel takes a median of **24.5 days**, rising by **8.5 days a year**, and reaches **40–42 days** in Tel Aviv and Jerusalem ¹.
Clalit's and Maccabi's online dermatology services explicitly exclude moles and suspected skin cancer.² These cases therefore require in-person assessment. In the absence of a dedicated pre-visit image-triage mechanism, the waiting list itself does not reveal which lesions are most urgent. A malignant lesion may therefore remain in the standard queue until it is reviewed by a clinician.

**What we build:** while waiting for their appointment, the patient uploads a smartphone photograph of the lesion and fills out a brief symptom form via the healthcare fund's app. The system scores every patient on the waiting list.
**[amended 30.10 — the original said "a short list of up to 10 suspicious cases", which fixed the list length regardless of queue size.]** Every morning, the system flags the highest-risk **10% of patients currently on the dermatologist's waiting list** for priority review. The number of flagged cases therefore scales with the size of the waiting list rather than being capped at ten: 5 out of 50, 10 out of 100, 30 out of 300. Each flagged case is shown with the photo, the answers and cited reasons.
For each case, the dermatologist decides whether to move the patient to an earlier in-person appointment and, where clinically and operationally appropriate, whether to issue a biopsy referral in advance to accelerate the diagnostic workup.

**Safety principle:** the system can only move a patient *earlier*. A patient who is not flagged keeps their regular appointment, so a miss costs at most today's situation. The patient never sees a risk score or a diagnosis, and the dermatologist decides on every case.

> **[amended 30.10 — this principle did not survive its own simulation, and that is a finding rather than a
> wording problem.]** The claim holds only if the urgent appointments are *added* to the clinic's capacity.
> The M1 simulation reallocates them instead — ten of the day's hundred appointments are reserved for
> priority review — and under that assumption the median wait for a benign lesion rises from 24 to **27
> days**. Patients who are not flagged are therefore delayed, which is the opposite of what this paragraph
> promises. A pilot that wants the safety principle to be true must fund **additional** slots, not
> reallocated ones; the cost of doing so is now quantified in §2 and `reports/m1_queue_simulation.json`.
> We state this rather than quietly restate the principle.

**AI-deletion test:** Without AI the problem remains: dozens of photos arrive every day, and no dermatologist can review them all in time to find the few that cannot wait.

## 2 · Success metrics (fixed before any code)

| # | Metric | Target | Outcome |
|---|---|---|---|
| M1 | Median simulated days to appointment for patients with malignant lesions (MEL + BCC + SCC), under the predefined waiting-list scenarios and fixed urgent-slot capacity | ≤ 7 days in the primary simulation scenario; always reported alongside the absolute and percentage reduction versus the FIFO baseline | **Met within the simulated fixed-capacity reallocation scenario** · 24.0 → **1.0 median days** (−96%). That scenario delays some non-prioritized patients and therefore does **not** validate the product's no-delay safety principle (§1) |
| M2 | Patient-level sensitivity for skin cancer (MEL + BCC + SCC → flagged for high-urgency dashboard triage) on held-out PAD-UFES-20 patients (real phone photos) | ≥ 0.90, reported with 95% confidence intervals and absolute TP/FN counts. Melanoma sensitivity is also reported separately. | **Not met** · **0.664** (95% CI 0.570–0.750) at the standard evaluation scenario; 0.820 at 15 slots |
| M3 | Groundedness on a frozen set of 20 guidance questions (5 unanswerable), plus a check that each number appears in its cited source | ≥ 0.9, and 5/5 refusals | **Met** · **0.933** groundedness after re-ranking and 15/15 numbers traced to their cited chunk. The 5/5 refusals were measured **without** re-ranking and carried across; see §8 CP2 |
| M4 | Agent scenario set of 15 complex workflow cases | ≥ 13/15 correct; zero cases in which the agent suppresses, downgrades or removes a patient who was flagged by the validated deterministic triage policy. | **Split** · accuracy **not met**: agent 9/15, agent behind the code guardrails 11/15, against **12/15 for a plain if-else script**. Safety **met**: no patient removed, no score or rank changed |

**Operating points [amended 26.10; clarified 30.10].** The charter originally gave the metrics without a
shared operating point, so M2 could have been reached by flagging almost everyone and M1 depended on an
unstated arrival rate. Both are now pinned. The product policy and the evaluation scenarios are stated
separately, because they are not the same thing:

> **Product policy:** flag the highest-risk **10% of patients currently on the full waiting list**.
>
> **M2's evaluation scenario:** a standardized **10 priority-review slots per 100 arrivals** at an assumed
> **10% malignancy prevalence**, applied to the frozen test patients.
>
> **M1's simulation:** a clinic with **100 daily arrivals, 100 daily appointments, and 10 of them reserved
> for priority review**, with the waiting list re-ranked every morning.

These evaluation scenarios represent a fixed daily review capacity. They are **not** identical to the
product's policy of flagging 10% of the entire waiting list, and no result below should be read as having
validated that policy directly.

The two calculations nevertheless agree closely. In the M1 simulation about **61%** of malignant patients
receive a priority appointment, against the separately measured validation M2 sensitivity of **0.612**. M2
is computed independently by `src/evaluate_test.py` on the held-out test patients and is **not** computed
inside the M1 simulation; the agreement is corroboration between two different methods on two different
splits, not one number appearing twice.

**What M1's median hides.** The wait distribution has no middle: a patient either reaches a priority slot
and is seen the next morning, or does not and waits the full queue. The median clears the 7-day target only
because more than half of malignant patients are flagged. At 5 slots only 41% are, the median returns to
25.0 days, and the policy becomes slightly *worse* than doing nothing. Four numbers belong beside the
median and are always reported with it: **44%** of malignant patients still wait more than 7 days, the 90th
percentile is **27 days** against 24 at baseline, the median wait for a benign lesion rises from 24 to **27
days**, and at 30 slots — where only 5% of cancer patients remain over target — a benign patient waits 34
days. The policy moves cancer patients forward by moving everyone else back.

**[amended 30.10]** Metrics are computed at the patient level, but not all on the same data. M2 is computed
on the held-out PAD-UFES-20 test patients, which carry patient IDs. M1 is a simulation built from the
**validation** patients' real model scores, because per-patient test scores were not retained and the test
set is opened once. M3 and M4 are measured on their own frozen evaluation sets, not on PAD at all. Training
sources without a patient ID are split at the lesion level; this is a limitation we state rather than hide.
For patients with multiple images, the aggregation rule is defined and frozen before test-set evaluation.

Sensitivity comes before list length: a missed melanoma costs far more than one extra case for the dermatologist to review. All metrics are stratified by patient age group, and by skin tone wherever it is recorded. Coverage is the limitation: PAD records Fitzpatrick for 65% of its images and holds 11 in V–VI, two of them in the test split; in the ISIC training pool 1,247 of 6,539 images carry Fitzpatrick, almost all from a single contributor. The simulation is run at 5, 10 and 15 urgent slots a day, and at 5%, 10% and 20% malignant prevalence, because the real mix is unknown.
Because the real-world prevalence is unknown, queue-level metrics are interpreted as scenario results rather than estimates of current healthcare-fund performance.

## 3 · Data (all four checks run 18–19.9; details in [`DATA_DECISIONS.md`](DATA_DECISIONS.md))
| Source | Role | Size | Why | Licence |
|---|---|---|---|---|
| **ISIC Archive**, clinical (non-dermoscopic) images, incl. MILK10k | **Main training data** |6,539 images, 470 melanomas, 761 nevi, 56.6% malignant (after removing PAD and the unlabelled benchmark; a further 124 Indeterminate rows — 123 actinic keratoses and one atypical melanocytic neoplasm — are dropped at preprocessing) | The largest pool of non-dermoscopic images with enough melanomas; the closest public proxy for a phone photo | CC-BY-NC / CC-BY (per image) |
| **PAD-UFES-20** | **Test set** of real phone photos; the only source with symptoms; training only from patients outside the test split |2,298 images, 1,891 lesions, 1,373 patients, **52 melanomas**, 47.4% malignant | The only source actually shot on smartphones, and the only one with the symptom answers our app collects. Too small to train on alone | CC BY 4.0 |
| ~~**HAM10000**~~ | ~~Pretraining only~~ **[amended 26.10: dropped]** | 10,015 images, 1,113 melanomas | **Rejected.** It overlaps ISIC, so pretraining on it risks the same lesion appearing in training and in evaluation, and it is dermoscopic: no image in it is a phone photo, so it cannot narrow the domain gap that actually matters. The final model is trained on ISIC clinical images and fine-tuned on PAD, with no HAM10000 stage. `reports/decisions.md` #8 | CC BY-NC 4.0 |
| **NCI PDQ** summaries | Guidance corpus for the RAG | 7 documents · 224 sections · 724 chunks **[amended 26.10: Health Professional versions, not patient summaries — see §8 CP1]** | Official, citable text for the reasons shown to the dermatologist | Free of copyright; credit NCI |

**Label policy (frozen).** The target is binary: MEL + BCC + SCC are malignant, everything else is not.
Actinic keratosis is pre-cancerous and is treated differently by the two sources, so the policy is stated
explicitly: ISIC classifies it as `Indeterminate` and we drop those 124 rows; PAD labels it `ACK` and we
keep it as a negative. The consequence is that roughly a third of the phone-photo test set is a lesion
class the image model never meets in training, and it is the most likely source of false alarms. We report
the false-alarm rate on that class separately rather than hide it inside the overall specificity.

**De-duplication rule (frozen)**. PAD appears twice: on Mendeley and inside the ISIC Archive as collection 406. The training pull is therefore defined as every ISIC image with image_type:"clinical: close-up" excluding collection 406 (PAD) and collection 424 (MILK10k Benchmark, whose labels are not released). That is 9,316 − 2,298 − 479 = 6,539 images, of which 470 are melanomas and 761 nevi. Without this exclusion every PAD image would appear in both train and test, and the melanoma count would be inflated from 470 to 522. The ISIC metadata export carries no collection field, so the exclusion is implemented on contributor attribution (PAD) and on a missing `diagnosis_1` (the benchmark). Both proxies were verified against collections 406 and 424 through the ISIC API on 15.10.2026: the id sets are identical, with zero differences in either direction.

**Plan B.** If the ISIC API or S3 is unavailable, MILK10k ships as a frozen challenge zip (MILK10k_Training_Input.zip, 314 MB, with its ground-truth and metadata CSVs), PAD-UFES-20 has an independent copy on Mendeley (doi:10.17632/zr7vgbcyr2.1), and HAM10000 has a third copy on Harvard Dataverse (doi:10.7910/DVN/DBW86T). These independent sources provide a fallback so no single host is a single point of failure.

**PAD split rule (frozen).** The PAD-UFES-20 patient split is created and frozen before model development. The symptom model, multimodal fusion/calibration layer, image-quality component, and all threshold or ranking-policy tuning use PAD training/validation patients only. Held-out PAD test patients are never used for model selection, calibration, threshold setting, or policy tuning.

**Rejected:**
- `BCN20000`: dermoscopic only, so beyond HAM10000 it adds volume and no new domain.
- `marmal88/skin_cancer`: a re-upload whose 13,354 rows hold only 10,015 unique images. **79.8% of its test images are also in train.**
- `Fitzpatrick 17k`: atlas photos, mostly non-cancerous conditions, with no age, sex or body site. Only **10 of 40 sampled image links still work**. Its advantage, diverse skin tones, does not survive the dead links.
- **Generative phone-to-dermoscopic conversion**: a generator would invent subsurface structures that the phone never captured. Unsafe for a medical product.
- **HAM10000** **[amended 26.10]**, for the reasons in the table above.

## 4 · Architecture

The system runs end-to-end as a multi-modal pipeline, utilizing an orchestrating agent to manage quality checks and contextual delivery to the clinician.

```mermaid
flowchart TD
    %% Data Input & Quality Routing
    A[Phone photo + symptom answers] --> B[Quality check + lesion crop]

    %% Error and Low-Confidence Routing
    B -. unusable photo / not a lesion / missing input .-> M[Manual review queue<br/>tool error · missing input]
    M --> F[Dermatologist:<br/>move earlier / approve referral draft / leave]

    %% Core Machine Learning Pipeline
    B --> C[Image model<br/>ConvNeXt-Tiny on ISIC clinical,<br/>fine-tuned on PAD train]
    A --> S[Symptom model<br/>trained on PAD train/validation only]

    %% Risk Fusion & Capacity Management
    C & S --> D[Fusion + patient-level risk score<br/>60% image rank + 40% symptom rank, nothing fitted]
    D --> L[Deterministic ranking policy<br/>top 10% of current waiting list]

    %% Knowledge & Explanation Layer
    VectorDB[(Vector DB)] <--> RAG[RAG over NCI PDQ<br/>Chroma DB + citations]
    L --> RAG
    RAG --> F

    %% Agent Supervision Layer
    G((Agent · Clinician Seat)) -. orchestrates tools and retrieval .-> B & C & S & RAG
    G -. on failure .-> M
  ```
It works end to end without the agent: both models score each patient, the deterministic ranking policy fills the day's priority slots, and each case receives a fixed model-evidence summary containing the relevant score bands, symptom flags and image-quality status. **[amended 30.10]** This is implemented and tested, not asserted: `USE_AGENT = False` produces the same report from the same ranking, and both versions are committed in `reports/agent/`.

### Baseline Evaluation Table
To justify the multi-modal design, the pipeline will be benchmarked against the following baselines (evaluated strictly on the same held-out PAD-UFES-20 test patients):

1. **Arrival order (today's practice):** every patient waits for their regular slot, a median of 24.5 days.¹ This is what M1 is measured against, and the only baseline that represents the current system.
2. **Metadata Only (The Floor):** A Logistic Regression model trained purely on patient age, lesion body site, and the six mandatory symptom questions. *~~AUC 0.883 on the 82% of rows where every mandatory app field is answered; letting missingness in as its own category reaches 0.907.~~* **[amended 22.10 — both figures superseded.]** Those two numbers were measured before the missingness policy was frozen, and the gap between them turned out to be the artefact rather than the signal: in PAD, `grew` and `changed` are unanswered for about 20% of lesions and 87–88% of those are malignant, because the history block was filled in mainly for patients who went on to biopsy. A model given *nothing but the count of missing optional fields* reaches AUC 0.839. The frozen policy is therefore to impute inside the pipeline and give no missingness indicator to any model, which costs accuracy and buys honesty. **The baseline is AUC 0.860 ± 0.021 in cross-validation and 0.867 on the frozen validation split.** The complete-case variant (0.893) is reported in `reports/results.md` §6 as a variant, not as the bar.
3. **Image Model Alone:** The vision component evaluated independently (fine-tuned on clinical images) to isolate the predictive power of visual features.
4. The Selected Integrated Triage Model (Multi-modal): Image Model + Symptom Model + fusion layer, evaluated against M1 and M2.
5. **[added 26.10]** **DINOv2 frozen + logistic regression** — the cheapest image baseline, no training at all. It was added because the expensive model has to beat more than a questionnaire.

The RAG explanation layer is evaluated separately under M3 and is not counted as part of predictive performance.

**Result, on the 275 held-out test patients (128 malignant), at M2's evaluation scenario:**

| Model | Patient AUC (95% CI) | Sensitivity | PPV |
|---|---|---|---|
| Arrival order — today's practice | 0.500 | 0.10 | 0.10 |
| Questionnaire only — the baseline | 0.862 (0.813–0.902) | 0.57 | 0.57 |
| Image model | 0.953 (0.929–0.972) | 0.69 | 0.69 |
| **Final: image 60% + questionnaire 40%** | **0.954** (0.928–0.975) | **0.66** | **0.68** |

Paired on the same patients: image − questionnaire **+0.092** (+0.052 to +0.136); fused − image **+0.000**
(−0.014 to +0.015). The full validation table, including DINOv2 and the ISIC-only model, is in
`README.md` §5 and `reports/results.md`.

## 5 · Agent Seat

**[amended 30.10 — the *decision* is restated. The original wording described orchestration, which a fixed
script can also do, and the charter feedback said so. The original *tool list* was close to correct and is
restored below; an earlier revision of this section wrongly reduced it to three.]**

- **Decision (what a fixed script cannot do):** the agent decides **only in the direction of escalation or
  information-gathering**, never downward. The risk score and the ranking are locked before it runs. Within
  that envelope it chooses between five outcomes — `proceed`, `proceed_with_flag`, `ask_patient`,
  `request_retake`, `manual_review` — and the judgement it is there for is: when a free-text note
  contradicts the form, which single follow-up question to ask; and when a photo is poor, whether a retake
  would actually help or whether the case should go forward flagged instead. It never sets, modifies or
  overrides the validated risk score or the deterministic ranking that fills the daily slots.
- **Structurally, not by instruction:** the schema the agent answers in (`CaseReview` in
  `agent/triage_agent.py`) has no score field and no rank field. There is nothing in its output for it to
  change them with.
- **Tools [amended 30.10 — aligned with the implementation]:** seven, registered in `build_crew()`:
  `get_case_data`, `predict_visual_risk`, `score_symptoms`, `check_image_quality`,
  `retrieve_guidance_context`, `request_retake` and `ask_patient`. The first five retrieve or inspect
  information that already exists and cannot change the score or the ranking; the image and symptom model
  outputs are computed upstream and exposed to the agent as read-only evidence. The last two prepare draft
  messages to the patient and send nothing: they queue a draft that plain code releases only after the
  guardrails pass, and they reject a draft containing reassuring wording or more than two sentences before
  it is ever queued. `draft_biopsy_referral` was listed on 4.10 and fell below the cut line; the code says
  so at the point where it would have been defined. The retrieval query is built by code from the symptoms
  answered yes and the body site rather than by the model, so retrieval is reproducible.
- **On failure:** retry once on malformed output, never retry a call that has hung, and after 120 seconds
  fall back to `manual_review` with the fixed model-evidence summary and the patient's priority unchanged.
  Every outcome, attempt count and duration is written to an agent log. Earlier runs with an exhausted quota
  and with a missing API key fell back for every patient and still produced a complete report; they are kept
  in `reports/agent/` as the evidence that the path works.
- **Guardrails** — enforced in code (`enforce_guardrails`), not in the prompt:
  * **The agent never issues a biopsy referral and never books an appointment on its own. It only drafts, and the dermatologist approves every action.**
  * Strict prohibition of "benign" or "safe" diagnostic wording in the dashboard or patient communications. Enforced in English and Hebrew, with Hebrew final letters normalised; quoted NCI passages are excluded from the check so a clinical sentence is not mistaken for reassurance.
  * The patient receives a neutral message ("your photo was received and will be reviewed") and never a risk estimate. A message leaves the system only for `request_retake` and `ask_patient`, after all checks pass.
  * Explicit "I don't know" fallback if clinical inquiries fall outside the verified NCI PDQ guidance corpus.
  * A citation survives only if its chunk id was retrieved **for that patient** and its quote appears word for word in that chunk. Every number in the summary must come from the case record or a surviving citation.
  * Patient-submitted text is treated as data, never as instructions.
  * Users under the age of 18 are automatically routed to direct clinical review regardless of AI score.

**M4, and the honest answer [added 30.10].** The 15 scenarios were built around the escalation decisions
above and an if-else script was run on the same 15, exactly as the charter feedback asked. The script scored
**12/15**, the agent **9/15**, and the agent behind the code guardrails **11/15**. By the test the feedback
itself proposed, *the agent is not needed for the decisions*: the rules do as well or better, and the
agent's decisions vary between runs of the same case. Where it did add judgement was scenario 9 — a lesion
behind the ear, where it saw that a retake would not help and proceeded with a flag while the script
demanded a retake. What it adds reliably is the patient-specific text: the question that targets the actual
gap in the form, the retake instruction in Hebrew, and the grounded citation. **The conclusion we draw is
to let code make every rule-shaped decision and leave the agent the texts and the cases the rules do not
cover.** `notebooks/08_agent_m4_vs_ifelse.ipynb`.

## 6 · Risks and cut line
| Risk | Mitigation |
|---|---|
| **Missing fields leak the label.** In PAD-UFES-20 the count of missing optional fields alone reaches **AUC 0.836** (development patients, patient-grouped CV), because those fields were recorded mainly for patients who went on to biopsy | Restrict training to the fields the app's mandatory flow always collects: age, body site and the six symptom questions. `biopsed` is banned by name, since it records that a biopsy took place |
| **Patient / lesion leakage.** Measured, not assumed: in PAD a naive image split leaks **29.3%** of test images (lesions keyed by `patient_id::lesion_id`, because 250 of the 1,641 raw ids are shared between patients). In the ISIC pool duplicate lesions are rare (0.2%), but 1,247 images come from only **79 patients**, all from one contributor | Group by patient wherever a patient id exists, by lesion where it does not. Splits are generated once, frozen, and reused by every experiment |
| **Prevalence mismatch** (**56.6%** of the ISIC training pool and **47.4%** of PAD are malignant, because both over-represent lesions selected for biopsy; the population rate is a few percent) | Use capacity-based ranking rather than a fixed probability threshold, and evaluate at 5%, 10% and 20% malignant prevalence. Report sensitivity at fixed review capacity. Prevalence shift remains an explicit limitation |
| **Data domain gap** (patients may submit photos with poor lighting, blur, framing problems or lens artefacts) | Build an image-quality dataset by manually labelling ~300 PAD images against a written usable/unusable rubric. Split these labels at the patient level; apply synthetic degradations (blur, under-/over-exposure, colour cast) only to the training portion, and evaluate the quality filter on held-out manually labelled patients. Measured at the training resolution: phone photos are **2.6× less sharp** than the largest ISIC contributor and **4.4×** less than the second, while brightness and contrast barely move — the gap is blur, not exposure |
| **Dark-skin performance cannot be evaluated with the current data.** The phone-photo test set holds **2** Fitzpatrick V–VI images out of 501, and PAD records skin tone for only 65% of its images (11 V–VI in total). ISIC holds **376** V–VI clinical close-ups, but all come from a single contributor and sit in the training pool, so a gap there would measure institution as much as skin tone | Report the V–VI slice as raw counts, never as a percentage; state in the README that this is unvalidated, and list a dark-skin evaluation set as the first requirement for any clinical pilot |
| **Melanoma-specific statistics are thin.** PAD-UFES-20 holds 52 melanomas in total, and the frozen test split contains **9 melanomas out of 501 images**, so a single miss moves sensitivity by 11 points | Report melanoma sensitivity with 95% confidence intervals and raw TP/FN counts. The headline M2 target applies to skin cancer overall (MEL + BCC + SCC); melanoma is reported separately and read cautiously |
| **[added 30.10] The field input is not the input the models were trained on.** Everyone on the waiting list gets a score, but the image model has only ever seen lesions, and many people wait for a rash, acne or a non-lesion complaint. A photo that is not a lesion could take a priority slot from a melanoma | Narrow intake to referrals for a mole or lesion, and route non-lesion photos out of scoring. **Partly built:** `not_a_lesion` is a recognised quality status and forces `manual_review` in code, so the routing exists and is enforced. The classifier that would *detect* it is not connected — `run_quality_check` currently reads the status from the daily-case record. Building and validating that classifier is the next step on this risk |
| **[added 30.10] The questionnaire was not filled in by patients.** In PAD the answers were recorded by senior medical students during a clinic visit (Pacheco et al., 2020). In the app the patient answers alone, so every questionnaire figure here is an upper bound | Measured rather than assumed: flipping 10% of the answers at random lowers patient AUC from 0.867 to **0.823** and the scenario sensitivity from 0.52 to **0.46**; over-reporting only ("no" → "yes") gives 0.835 and 0.49. Reported in `reports/results.md` §6 and stated as an upper bound wherever a questionnaire number appears |

**Cut line:** *If only two weeks remain, we drop the biopsy-referral draft and advanced agent workflows. We keep a minimal agent that invokes the quality check, retrieves grounded guidance, and assembles clinician-facing evidence. The core triage pipeline remains deterministic: smartphone photo + symptom form → patient-level risk score → ranked waiting list → daily priority list.*

**[30.10: the cut line was used.** `draft_biopsy_referral` was dropped and is not implemented. Everything
else on the near side of the line is built.**]**

## 7 · Milestones and hats
| Date | Deliverable |
|---|---|
| 4.10 | Finalized Product Charter & Data Verification |
| **18.10 · CP1** | Core End-to-End Pipeline (No Agent); Baseline Evaluation incl. re-measured metadata floor; M3 Groundedness Benchmark |
| **30.10 · CP2** | Agent Orchestration Layer Integrated; M4 Validation; 1-Minute Live Demo |
| **31.10 · CP3** | Validation Prediction-to-Agent Integration; Real PAD Patient Cases; End-to-End Report Generation and Verification |
| **1.11** | Repo frozen ahead of the mentoring session |
| **4.11** | Individual mentoring session (working system + draft deck) |
| 12.11 | Repository cleanup, final evaluation, and stakeholder presentation deck |
| **15.11 / 18.11** | Project presentations |

| Hat | Owner | Responsibility |
|---|---|---|
| Data | Yuval Rubenchuk | Pipeline architecture, strict patient-level splitting, data cleaning, and leakage mitigation |
| Model | Roi Budnitsky | Training baselines, model fine-tuning, threshold tuning, and multi-ethnic performance analysis |
| Agent | Tami Khazma | Agent tool building, orchestration loops, safety guardrails, and M4 validation set execution |
| Product | Ifat Davidson | Clinician workflow UX, patient-facing safety copy, README documentation, and final fund presentation deck |

## 8 · Checkpoint updates

### CP1 · 18.10 — core pipeline, no agent

**1. What works end to end.** ISIC acquisition and audit, the frozen PAD patient split, preprocessing, and
the questionnaire baseline: `notebooks/01_eda_isic.ipynb` → `04_symptom_model.ipynb`, all running top to
bottom with their outputs committed.

**2. Metric against target.** M2's bar was set: the questionnaire alone reaches **patient AUC 0.867** on the
frozen validation split without a photograph. M3's corpus and metric were built and the benchmark run
(`notebooks/07_rag_nci_m3.ipynb`): **groundedness 0.733 before re-ranking, 5/5 refusals**. No image model yet.

**3. Decided since the charter, and why.** The missingness policy was frozen after measuring that the count
of missing optional fields alone reaches AUC 0.839 — the charter's 0.883/0.907 pair was measuring that
artefact, so both were retired (§4). The NCI corpus was switched from the patient summaries to the **Health
Professional** versions, in answer to the charter feedback's question about what a dermatologist learns from
a summary written for patients. A naive image-level split was measured to leak 29.3% of PAD test images, so
patient-level grouping became an assertion in every notebook rather than a printed report.

**4. Stuck, and what would release it.** The image model had no training run yet, so the central question —
how much the photograph adds over the questionnaire — was still open.

**5. Next three steps.** Train the image model on ISIC and fine-tune on PAD · fuse it with the questionnaire
and measure the paired difference · build the agent's tools and guardrails in code.

### CP2 · 30.10 — agent integrated

**1. What works end to end.** The image and questionnaire models, the fusion and the deterministic ranking
pipeline are implemented and evaluated, ending in a single held-out test run
(`notebooks/01`–`06`, `src/`). The agent layer is implemented and demonstrated on a synthetic daily list of
ten records using the expected daily-case interface (`agent/`, `notebooks/07`–`09`): it produces
clinician-facing evidence reports, applies its guardrails in code, retrieves and cites grounded guidance,
and supports a non-agent fallback path.

**The full handoff from live model predictions into the agent's daily-case input has not yet been
demonstrated as one integrated execution.** The daily list the agent runs on is made up
(`data/processed/daily_cases_sample.csv`, patient ids `P-0193` and the like, not PAD ids), and
`agent/make_daily_list.py` — the function that would build it from scored patients — has not been run on
real model output. The repository therefore contains the implemented components and their separate
evaluations; joining them into one execution is the next step, and the interface both sides already agree
on is the daily-case CSV.

A second gap belongs here rather than in a footnote: the committed output of
`notebooks/09_agent_daily_report.ipynb` is from a run of the **non-agent** path — all ten patients show
`model_evidence_only`. The agent run of the same list exists and is committed as
`reports/agent/triage_report_sample_final.docx` and its log, where all ten show `"outcome": "agent"` with no
fallback, but the notebook as saved does not show it. It should be re-run with `USE_AGENT = True` and
re-committed.

**2. Metrics against targets.** **M1 met within its simulated scenario** — 24.0 → 1.0 median days, with the
reallocation caveat in §1 and §2. **M2 not met** — 0.664 (CI 0.570–0.750) against a target of 0.90, on 275
held-out patients. **M3 met** — 0.933 groundedness after re-ranking; the 5/5 refusal test was run **before**
re-ranking and `reports/agent/m3_summary.csv` carries that same figure into the "after" row, so the refusal
result under re-ranking is inherited rather than measured. Re-running that one cell would settle it.
**M4 split** — 9/15 for the agent and 11/15 for the system against 12/15 for an if-else script, so not met
on accuracy; met in full on safety.

**3. Decided since CP1, and why.** The 60/40 fusion was chosen before any result and kept: it ties the image
model rather than beating it (+0.000, CI −0.014 to +0.015 on test), and the reason recorded for it was
robustness when a photograph is poor, not performance. Doubling the SCC class weight was adopted on its
stated criterion; it cost melanoma on average (2.7 → 2.0 of 6 flagged across seeds) and that cost is
reported. Seed averaging was rejected: +0.0006, CI −0.007 to +0.008. The test set was opened **once**, on
30.10, on decisions frozen and team-approved on 26.10 in `reports/decisions.md`; `notebooks/06_test_run.ipynb`
is the executed record, including the dress rehearsal on validation that ran first. The M1 simulation
produced the finding that the product's safety principle does not hold under reallocated capacity (§1).

**4. Stuck, and what would release it.** Nothing is blocked. Four things are unfinished and listed above or
below: the end-to-end handoff, notebook 09's committed run, the M3 refusal re-measurement, and the
surgical-marker ink check, whose blind labelling sheet was written by the test run and awaits 498 manual
labels.

**5. Next three steps.** Run `make_daily_list.py` on the real scored patients and re-run notebook 09 with the
agent, closing the integration gap · label the ink sheet and run `src/ink_effect.py` · build the 10-slide
deck and the one-minute demo recording.

**Known gaps at CP2, stated rather than discovered later.** M2 and M4 are not met and are reported with the
numbers they reached and a diagnosis for each. The image-quality classifier is not connected (§6). One test
patient (PAT_759) is scored benign because PAD stores the same photograph under two lesion ids with
contradictory diagnoses; the de-duplication keeps the benign row, so the test set evaluates 128 malignant
patients rather than the 129 in the frozen manifest — recorded in `reports/decisions.md` before the run. The
direction of that error is argued rather than measured: the photograph carries a BCC label in one row and
a benign label in the kept row, so a model that scores it highly is charged a false alarm, which can lower
the reported figure but cannot raise it. The git history carries commits from two of the four team members;
the other two contributed through files merged under those accounts.


### CP3 · 31.10 — validation predictions connected to the Agent

**1. What works end to end.**

The integration gap identified at CP2 has been closed for the validation demonstration.

`src/build_daily_cases_from_val.py` now connects previously computed PAD validation predictions to the Agent's daily-case input.

The demonstration follows this pipeline:

Stored image and questionnaire predictions → 60/40 rank fusion → priority list → image-quality annotations → CrewAI Agent → dermatologist Word report.

The starting point is 220 PAD validation patients (103 malignant). To ensure that each image, symptom form and model score refer to the same lesion, the demonstration is restricted to 138 eligible patients with exactly one lesion and one photograph (47 malignant).

A seeded simulation selects 100 candidates, including 10 malignant cases by design. The frozen 60/40 percentile-rank fusion ranks these candidates and selects the 10 highest-priority patients.

Six of the ten malignant cases were included in the priority list.

Ground-truth diagnoses are stored separately in `data/processed/val_100_candidates_audit.csv` for evaluation purposes. They are never included in the Agent's input.

**2. Agent execution and verification.**

The ten selected patients were processed in `notebooks/09_agent_daily_report.ipynb` with `USE_AGENT = True`.

Results of the final committed run:

| Check | Result |
|---|---|
| Patients selected for priority review | 10 |
| Successful Agent executions | 10/10 |
| Retries | 0 |
| Fallbacks | 0 |
| `proceed` decisions | 8 |
| `proceed_with_flag` decisions | 2 |
| Guardrail corrections | 0 |
| Citations retained | 10 |
| Distinct cited source chunks | 3 |
| Patient ranking preserved | Yes |
| Word report generated | Yes |
| Regression tests | 4 passed |

Image-quality annotations were produced through an AI-assisted visual review of screenshots: 8 images were labelled `usable` and 2 were labelled `borderline`, both with `glare`.

These annotations are demonstration inputs, not outputs of a validated image-quality classifier or independent dermatologist assessment.

The Agent's two `proceed_with_flag` decisions followed the existing policy: borderline image quality does not automatically delay a patient in the `very high` priority band.

The Word report was verified to include all ten patients in the original model-ranking order. The committed execution log matches the final notebook output.

**3. Committed evidence.**

The reproducible integration and its outputs are documented in:

- `src/build_daily_cases_from_val.py` — validation prediction-to-Agent bridge.
- `data/processed/daily_cases_val.csv` — ranked Agent input.
- `data/processed/val_100_candidates_audit.csv` — candidate evaluation audit, including diagnostic labels.
- `data/processed/val_top10_quality_template.csv` — image-quality annotations.
- `notebooks/09_agent_daily_report.ipynb` — executed integration notebook and verification checks.
- `reports/agent/triage_report_validation_20261031.docx` — final dermatologist report.
- `reports/agent/triage_report_validation_20261031_agent_log.json` — matching Agent execution log.
- `tests/test_risk_bands.py` — regression tests for the shared priority-band policy.

The original non-Agent report-generation path also remains available and was tested separately.

**4. Metrics and decisions since CP2.**

The integration work did not change the frozen model, evaluation methodology or held-out test results.

- **M1:** Met its simulated median waiting-time target (24.0 to 1.0 days), while some patients experience longer waits because priority appointments are reallocated from existing capacity.
- **M2:** Not met. Held-out test sensitivity remains 0.664 (95% CI 0.570–0.750), below the target of 0.90.
- **M3:** Citation-ID groundedness reached 0.933 after re-ranking. The 5/5 out-of-corpus refusals were measured before re-ranking; refusal behavior after re-ranking has not been independently re-tested.
- **M4:** Not met on decision accuracy. The Agent scored 9/15, the system with guardrails scored 11/15, and the if-else baseline scored 12/15. The tested ranking-preservation safeguards held.

The integration confirms technical execution, not improved clinical sensitivity.

**5. Remaining limitations and next steps.**

The CP3 demonstration is complete for the specific stored-prediction-to-Agent handoff, but it is not a full prospective clinical workflow.

Remaining limitations:

- The final demonstration used stored model predictions rather than executing fresh image inference from the original photographs.
- The 100-candidate queue was sampled to contain 10 malignant cases and is not representative of real clinical prevalence.
- Only single-lesion, single-photo patients were eligible.
- Four of the ten malignant cases were not selected for priority review in the demonstration.
- Automated image-quality assessment and non-lesion detection are not yet connected.
- Risk scores and priority bands are queue-relative ranking measures, not calibrated probabilities of malignancy.
- M3 refusal behavior after re-ranking remains to be re-evaluated.
- The blind surgical-marker ink analysis remains unfinished; 498 test photographs await manual annotation.
- Production costs and clinical generalization have not been validated.

**Next steps:** complete the remaining documentation and presentation materials; evaluate the unresolved data and Agent limitations; and, if time permits, demonstrate fresh photograph-to-report inference without changing the frozen test results.

**CP3 conclusion.**

The project now demonstrates a reproducible technical connection between stored validation predictions, deterministic patient prioritization, Agent-generated evidence and a clinician-facing report.

This closes the integration gap documented at CP2 while preserving the distinction between technical success and clinical validation.


---
¹ Murad H et al. *Measuring geographical disparities in waiting times for community-based specialist care.* Israel Journal of Health Policy Research, 2025. [doi:10.1186/s13584-025-00702-7](https://doi.org/10.1186/s13584-025-00702-7). Covers all dermatologists in all four health funds, 2019–2023.

² Maccabi, [online dermatologist consultation](https://www.maccabi4u.co.il/31276/digital-services/communication/consultation_dermatologists/): "not intended for urgent cases or for diagnosing moles and skin lesions". Clalit, [online dermatologists](https://www.clalit.co.il/he/online_doctors/Pages/dermatologist_on_line.aspx): excludes "diagnosis and assessment of nevi" and suspected skin cancer. Both checked 18.9.2026.
