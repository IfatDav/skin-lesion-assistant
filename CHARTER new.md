# Project Charter · Skin Lesion Triage for Healthcare Providers

**Team:** Ifat Davidson · Tami Khazma · Roi Budnitsky · Yuval Rubenchuk | **Course:** BIU DS23 Capstone | **Date:** 04.10.2026 · updated 29.10.2026 (sections 3–5 aligned with the agent as built)
**Retrieval asset:** the module 5 grounded assistant.

## 1 · Problem & User
**User:** A dermatologist working within an Israeli healthcare fund (e.g., Clalit, Maccabi) who manages an overloaded appointment queue.

**Pain:**  the dermatologist cannot tell which patient on the waiting list cannot wait.
A public-system dermatologist appointment in Israel takes a median of **24.5 days**, rising by **8.5 days a year**, and reaches **40–42 days** in Tel Aviv and Jerusalem ¹.
Clalit's and Maccabi's online dermatology services explicitly exclude moles and suspected skin cancer.² These cases therefore require in-person assessment. In the absence of a dedicated pre-visit image-triage mechanism, the waiting list itself does not reveal which lesions are most urgent. A malignant lesion may therefore remain in the standard queue until it is reviewed by a clinician.

**What we build:** while waiting for their appointment, the patient uploads a smartphone photograph of the lesion and fills out a brief symptom form via the healthcare fund's app. The system scores every patient on the waiting list. Every morning it shows the dermatologist a **short list of up to 10 suspicious cases**, each with the photo, the answers and cited reasons.
For each case, the dermatologist decides whether to move the patient to an earlier in-person appointment and, where clinically and operationally appropriate, whether to issue a biopsy referral in advance to accelerate the diagnostic workup.

**Safety principle:** the system can only move a patient *earlier*. A patient who is not flagged keeps their regular appointment, so a miss costs at most today's situation. The patient never sees a risk score or a diagnosis, and the dermatologist decides on every case.
  
**AI-deletion test:** Without AI the problem remains: dozens of photos arrive every day, and no dermatologist can review them all in time to find the few that cannot wait.

## 2 · Success metrics (fixed before any code)

| # | Metric | Target |
|---|---|---|
| M1 | Median simulated days to appointment for patients with malignant lesions (MEL + BCC + SCC), under the predefined waiting-list scenarios and fixed urgent-slot capacity | ≤ 7 days in the primary simulation scenario; always reported alongside the absolute and percentage reduction versus the FIFO baseline |
| M2 | Patient-level sensitivity for skin cancer (MEL + BCC + SCC → flagged for high-urgency dashboard triage) on held-out PAD-UFES-20 patients (real phone photos) | ≥ 0.90, reported with 95% confidence intervals and absolute TP/FN counts. Melanoma sensitivity is also reported separately. |
| M3 | Groundedness on a frozen set of 20 guidance questions (5 unanswerable), plus a check that each number appears in its cited source | ≥ 0.9, and 5/5 refusals |
| M4 | Agent scenario set of 15 complex workflow cases | ≥ 13/15 correct; zero cases in which the agent suppresses, downgrades or removes a patient who was flagged by the validated deterministic triage policy. |

All reported metrics are computed at the patient level on the PAD-UFES-20 test set, which carries patient IDs. Training sources without a patient ID are split at the lesion level instead; this is a limitation we state rather than hide. For patients with multiple images, the aggregation rule is defined and frozen before test-set evaluation.

Sensitivity comes before list length: a missed melanoma costs far more than one extra case for the dermatologist to review. All metrics are stratified by patient age group, and by skin tone wherever it is recorded. Coverage is the limitation: PAD records Fitzpatrick for 65% of its images and holds 11 in V–VI, two of them in the test split; in the ISIC training pool 1,247 of 6,539 images carry Fitzpatrick, almost all from a single contributor. The simulation is run at 5, 10 and 15 urgent slots a day, and at 5%, 10% and 20% malignant prevalence, because the real mix is unknown.
Because the real-world prevalence is unknown, queue-level metrics are interpreted as scenario results rather than estimates of current healthcare-fund performance.

## 3 · Data (all four checks run 18–19.9; details in [`docs/DATA_DECISIONS.md`](docs/DATA_DECISIONS.md))
| Source | Role | Size | Why | Licence |
|---|---|---|---|---|
| **ISIC Archive**, clinical (non-dermoscopic) images, incl. MILK10k | **Main training data** |6,539 images, 470 melanomas, 761 nevi, 56.6% malignant (after removing PAD and the unlabelled benchmark; a further 124 Indeterminate rows — 123 actinic keratoses and one atypical melanocytic neoplasm — are dropped at preprocessing) | The largest pool of non-dermoscopic images with enough melanomas; the closest public proxy for a phone photo | CC-BY-NC / CC-BY (per image) |
| **PAD-UFES-20** | **Test set** of real phone photos; the only source with symptoms; training only from patients outside the test split |2,298 images, 1,891 lesions, 1,373 patients, **52 melanomas**, 47.4% malignant | The only source actually shot on smartphones, and the only one with the symptom answers our app collects. Too small to train on alone | CC BY 4.0 |
| **HAM10000** | **Pretraining only** | 10,015 images, 1,113 melanomas | Dermoscopic (10× magnification, polarised light), so it shows subsurface structures a phone cannot capture. Useful as a starting point for the network, never as the target domain | CC BY-NC 4.0 |
| **NCI PDQ** Health Professional summaries | Guidance corpus for the RAG | 7 documents (screening, prevention, melanoma treatment, skin cancer treatment, genetics, childhood melanoma, childhood BCC/SCC); 724 chunks | Official, citable text written for clinicians, for the reasons shown to the dermatologist | Free of copyright; credit NCI |

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

## 4 · Architecture

The system runs end-to-end as a multi-modal pipeline, utilizing an orchestrating agent to manage quality checks and contextual delivery to the clinician.

```mermaid
flowchart TD
    %% Data Input & Quality Routing
    A[Phone photo + symptom answers] --> B[Quality check + lesion crop]

    %% Error and Low-Confidence Routing
    B -. unusable photo / missing input .-> M[Manual review queue<br/>tool error · missing input]
    M --> F[Dermatologist:<br/>move earlier / leave]

    %% Core Machine Learning Pipeline
    B --> C[Image model<br/>pretrained on HAM10000,<br/>fine-tuned on ISIC clinical]
    A --> S[Symptom model<br/>trained on PAD train/validation only]

    %% Risk Fusion & Capacity Management
    C & S --> D[Fusion + patient-level risk score<br/>trained/calibrated on PAD train/validation only]
    D --> L[Deterministic ranking policy<br/>daily top-10 within urgent-slot capacity]

    %% Knowledge & Explanation Layer
    VectorDB[(Vector DB)] <--> RAG[RAG over NCI PDQ<br/>Chroma DB + citations]
    L --> RAG
    RAG --> F

    %% Agent Supervision Layer
    G((Agent · Clinician Seat)) -. orchestrates tools and retrieval .-> B & C & S & RAG
    G -. on failure .-> M
  ```
It works end to end without the agent: both models score each patient, the deterministic ranking policy fills the day's urgent slots, and each case receives a fixed model-evidence summary containing the relevant score bands, symptom flags and image-quality status.

### Baseline Evaluation Table
To justify the multi-modal design, the pipeline will be benchmarked against the following baselines (evaluated strictly on the same held-out PAD-UFES-20 test patients):

1. **Arrival order (today's practice):** every patient waits for their regular slot, a median of 24.5 days.¹ This is what M1 is measured against, and the only baseline that represents the current system.
2. **Metadata Only (The Floor):** A Logistic Regression model trained purely on patient age, lesion body site, and the six mandatory symptom questions. *Re-measured on development patients only, with patient-grouped stratified cross-validation: **AUC 0.883** on the 82% of rows where every mandatory app field is answered. Letting missingness in as its own category reaches 0.907, which is why the complete-case number is the one the image and fusion models must beat.*
3. **Image Model Alone:** The vision component evaluated independently (fine-tuned on clinical images) to isolate the predictive power of visual features.
4. The Selected Integrated Triage Model (Multi-modal): Image Model + Symptom Model + fusion layer, evaluated against M1 and M2.
The RAG explanation layer is evaluated separately under M3 and is not counted as part of predictive performance.

## 5 · Agent Seat
- **Decision:** for each case the agent orchestrates the available tools, writes the patient-specific retake instruction or follow-up question, weighs the cases the rules do not settle, retrieves the most relevant verified guideline passages, and assembles the evidence shown to the dermatologist. When critical information is missing or a tool fails, it routes the case to manual review. **It never sets, modifies or overrides the validated risk score or the deterministic ranking that fills the daily slots.** Rules with a clear answer (age, missing fields, not a lesion, a photo a retake can fix) are enforced in code: in M4 a rule script matched or beat the agent on decisions alone.
- **Tools:** `get_case_data`, `check_image_quality`, `predict_visual_risk`, `score_symptoms`, `retrieve_guidance_context`, `request_retake`, `ask_patient`. `draft_biopsy_referral` was dropped under the cut line (section 6).
- On failure: Fall back to the non-agent path. Any tool error, timeout (120 s), malformed output after one retry, or missing critical input routes the case to a separate manual-review queue and never lowers its validated model priority or removes its regular appointment.
- **Guardrails:**
  * **The agent never issues a biopsy referral and never books an appointment. Its only patient messages are drafts (a retake instruction or one follow-up question), sent after the code checks; the dermatologist decides every action.**
  * Strict prohibition of "benign" or "safe" diagnostic wording in the dashboard or patient communications.
  * The patient receives a neutral message ("your photo was received and will be reviewed"; app copy, Product hat) and never a risk estimate.
  * Explicit "I don't know" fallback if clinical inquiries fall outside the verified NCI PDQ guidance corpus.
  * Patient-submitted text is treated as data, never as instructions.
  * Users under the age of 18 are automatically routed to direct clinical review regardless of AI score.

## 6 · Risks and cut line
| Risk | Mitigation |
|---|---|
| **Missing fields leak the label.** In PAD-UFES-20 the count of missing optional fields alone reaches **AUC 0.836** (development patients, patient-grouped CV), because those fields were recorded mainly for patients who went on to biopsy | Restrict training to the fields the app's mandatory flow always collects: age, body site and the six symptom questions. `biopsed` is banned by name, since it records that a biopsy took place |
| **Patient / lesion leakage.** Measured, not assumed: in PAD a naive image split leaks **29.3%** of test images (lesions keyed by `patient_id::lesion_id`, because 250 of the 1,641 raw ids are shared between patients). In the ISIC pool duplicate lesions are rare (0.2%), but 1,247 images come from only **79 patients**, all from one contributor | Group by patient wherever a patient id exists, by lesion where it does not. Splits are generated once, frozen, and reused by every experiment |
| **Prevalence mismatch** (**56.6%** of the ISIC training pool and **47.4%** of PAD are malignant, because both over-represent lesions selected for biopsy; the population rate is a few percent) | Use capacity-based ranking rather than a fixed probability threshold, and evaluate at 5%, 10% and 20% malignant prevalence. Report sensitivity at fixed review capacity. Prevalence shift remains an explicit limitation |
| **Data domain gap** (patients may submit photos with poor lighting, blur, framing problems or lens artefacts) | Build an image-quality dataset by manually labelling ~300 PAD images against a written usable/unusable rubric. Split these labels at the patient level; apply synthetic degradations (blur, under-/over-exposure, colour cast) only to the training portion, and evaluate the quality filter on held-out manually labelled patients. |
| **Dark-skin performance cannot be evaluated with the current data.** The phone-photo test set holds **2** Fitzpatrick V–VI images out of 501, and PAD records skin tone for only 65% of its images (11 V–VI in total). ISIC holds **376** V–VI clinical close-ups, but all come from a single contributor and sit in the training pool, so a gap there would measure institution as much as skin tone | Report the V–VI slice as raw counts, never as a percentage; state in the README that this is unvalidated, and list a dark-skin evaluation set as the first requirement for any clinical pilot |
| **Melanoma-specific statistics are thin.** PAD-UFES-20 holds 52 melanomas in total, and the frozen test split contains **9 melanomas out of 501 images**, so a single miss moves sensitivity by 11 points | Report melanoma sensitivity with 95% confidence intervals and raw TP/FN counts. The headline M2 target applies to skin cancer overall (MEL + BCC + SCC); melanoma is reported separately and read cautiously |

**Cut line:** *If only two weeks remain, we drop the biopsy-referral draft and advanced agent workflows. We keep a minimal agent that invokes the quality check, retrieves grounded guidance, and assembles clinician-facing evidence. The core triage pipeline remains deterministic: smartphone photo + symptom form → patient-level risk score → ranked waiting list → daily top-10.*

## 7 · Milestones and hats
| Date | Deliverable |
|---|---|
| 4.10 | Finalized Product Charter & Data Verification |
| **18.10 · CP1** | Core End-to-End Pipeline (No Agent); Baseline Evaluation incl. re-measured metadata floor; M3 Groundedness Benchmark |
| **30.10 · CP2** | Agent Orchestration Layer Integrated; M4 Validation; 1-Minute Live Demo |
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

---
¹ Murad H et al. *Measuring geographical disparities in waiting times for community-based specialist care.* Israel Journal of Health Policy Research, 2025. [doi:10.1186/s13584-025-00702-7](https://doi.org/10.1186/s13584-025-00702-7). Covers all dermatologists in all four health funds, 2019–2023.

² Maccabi, [online dermatologist consultation](https://www.maccabi4u.co.il/31276/digital-services/communication/consultation_dermatologists/): "not intended for urgent cases or for diagnosing moles and skin lesions". Clalit, [online dermatologists](https://www.clalit.co.il/he/online_doctors/Pages/dermatologist_on_line.aspx): excludes "diagnosis and assessment of nevi" and suspected skin cancer. Both checked 18.9.2026.
