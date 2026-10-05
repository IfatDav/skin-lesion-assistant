# Project Charter · Skin Lesion Triage for Healthcare Providers

**Team:** Ifat Davidson · Tami Khazma · Roi Budnitsky · Yuval Rubenchuk | **Course:** BIU DS23 Capstone | **Date:** 04.10.2026
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

Sensitivity comes before list length: a missed melanoma costs far more than one extra case for the dermatologist to review. All metrics are stratified by skin tone (the PAD Fitzpatrick field and the MILK10k skin-tone field) and by patient age group. The simulation is run at 5, 10 and 15 urgent slots a day, and at 5%, 10% and 20% malignant prevalence, because the real mix is unknown.
Because the real-world prevalence is unknown, queue-level metrics are interpreted as scenario results rather than estimates of current healthcare-fund performance.

## 3 · Data (all four checks run 18–19.9; details in [`docs/DATA_DECISIONS.md`](docs/DATA_DECISIONS.md))
| Source | Role | Size | Why | Licence |
|---|---|---|---|---|
| **ISIC Archive**, clinical (non-dermoscopic) images, incl. MILK10k | **Main training data** | 6,539 images, 470 melanomas, 761 nevi (after removing PAD and the unlabelled benchmark) | The largest pool of non-dermoscopic images with enough melanomas; the closest public proxy for a phone photo | CC-BY-NC / CC-BY (per image) |
| **PAD-UFES-20** | **Test set** of real phone photos; the only source with symptoms; training only from patients outside the test split | 2,298 images, 1,373 patients, **52 melanomas** | The only source actually shot on smartphones, and the only one with the symptom answers our app collects. Too small to train on alone | CC BY 4.0 |
| **HAM10000** | **Pretraining only** | 10,015 images, 1,113 melanomas | Dermoscopic (10× magnification, polarised light), so it shows subsurface structures a phone cannot capture. Useful as a starting point for the network, never as the target domain | CC BY-NC 4.0 |
| **NCI PDQ** patient summaries | Guidance corpus for the RAG | ~10–20 documents | Official, citable text for the reasons shown to the dermatologist | Free of copyright; credit NCI |

**De-duplication rule (frozen). PAD appears twice: on Mendeley and inside the ISIC Archive as collection 406. The training pull is therefore defined as every ISIC image with image_type:"clinical: close-up" excluding collection 406 (PAD) and collection 424 (MILK10k Benchmark, whose labels are not released). That is 9,316 − 2,298 − 479 = 6,539 images, of which 470 are melanomas and 761 nevi. Without this exclusion every PAD image would appear in both train and test, and the melanoma count would be inflated from 470 to 522.

**Plan B.** If the ISIC API or S3 is unavailable, MILK10k ships as a frozen challenge zip (MILK10k_Training_Input.zip, 314 MB, with its ground-truth and metadata CSVs), PAD-UFES-20 has an independent copy on Mendeley (doi:10.17632/zr7vgbcyr2.1), and HAM10000 has a third copy on Harvard Dataverse (doi:10.7910/DVN/DBW86T). These independent sources provide a fallback so no single host is a single point of failure.

**PAD split rule (frozen).** The PAD-UFES-20 patient split is created and frozen before model development. The symptom model, multimodal fusion/calibration layer, image-quality component, and all threshold or ranking-policy tuning use PAD training/validation patients only. Held-out PAD test patients are never used for model selection, calibration, threshold setting, or policy tuning.

**Rejected:**
- `BCN20000`: dermoscopic only, so beyond HAM10000 it adds volume and no new domain.
- `marmal88/skin_cancer`: a re-upload whose 13,354 rows hold only 10,015 unique images. **79.8% of its test images are also in train.**
- `Fitzpatrick 17k`: atlas photos, mostly non-cancerous conditions, with no age, sex or body site. Only **10 of 40 sampled image links still work**. Its advantage, diverse skin tones, does not survive the dead links.
- **Generative phone-to-dermoscopic conversion**: a generator would invent subsurface structures that the phone never captured. Unsafe for a medical product.

**Skin-tone reporting without DDI:** M2 is stratified by the Fitzpatrick field in PAD-UFES-20 (our test set) and the MILK10k skin-tone field (0–5) inside ISIC. Both skew light, so performance on Fitzpatrick V–VI stays an **open limitation** that we state rather than hide.

## 4 · Architecture

The system runs end-to-end as a multi-modal pipeline, utilizing an orchestrating agent to manage quality checks and contextual delivery to the clinician.

```mermaid
flowchart TD
    %% Data Input & Quality Routing
    A[Phone photo + symptom answers] --> B[Quality check + lesion crop]
    
    %% Error and Low-Confidence Routing (Fail-safe)
    B -. unusable photo / low confidence .-> M[Manual review queue<br/>tool error · missing input]
    M --> F[Dermatologist:<br/>move earlier / pre-issue biopsy / leave]

    %% Core Machine Learning Pipeline
    B --> C[Image model<br/>pretrained on HAM10000,<br/>fine-tuned on ISIC clinical]
    A --> S[Symptom model<br/>trained on PAD-UFES-20]
    
    %% NEW: Explainability Verification (AI-Deletion Test)
    C --> XAI[AI-deletion test<br/>Explainability Validation]

    %% Knowledge Base Layer (Fixed Position)
    VectorDB[(Vector DB)] <--> RAG[RAG over NCI PDQ<br/>Chroma DB + citations]

    %% Core Logic & Capacity Management
    XAI & S & RAG --> D[Risk score per patient]
    D --> L[Daily top-10 list<br/>within urgent-slot capacity]
    L --> F

    %% Human-in-the-Loop Feedback Loop (Continuous Learning)
    F -.-> |Model improvement feedback| C & S

    %% Agent Supervision Layer
    G((Agent · Clinician Seat)) -. orchestrates .-> B & C & S & RAG & D
    G -. on failure .-> M
  ```
It works end to end without the agent: both models score each patient, the deterministic ranking policy fills the day's urgent slots, and each case receives a fixed model-evidence summary containing the relevant score bands, symptom flags and image-quality status.

### Baseline Evaluation Table
To justify the multi-modal design, the pipeline will be benchmarked against the following baselines (evaluated strictly on the same held-out PAD-UFES-20 test patients):

1. **Arrival order (today's practice):** every patient waits for their regular slot, a median of 24.5 days.¹ This is what M1 is measured against, and the only baseline that represents the current system.
2. **Metadata Only (The Floor):** A standard Logistic Regression model trained purely on patient age, sex, and the symptom checklist. *EDA put this at AUC 0.90, but that figure was measured before the missing-field artefact below was controlled for. It is therefore treated as provisional: the bar is re-established at CP1 using only the fields the app's mandatory flow collects, and the re-measured number is what the image and fusion models must beat.*
3. **Image Model Alone:** The vision component evaluated independently (fine-tuned on clinical images) to isolate the predictive power of visual features.
4. The Selected Integrated Triage Model (Multi-modal): Image Model + Symptom Model + fusion layer, evaluated against M1 and M2.
The RAG explanation layer is evaluated separately under M3 and is not counted as part of predictive performance.

## 5 · Agent Seat
- **Decision (what a fixed script cannot do): for each case the agent orchestrates the available tools, handles photo-quality and missing-information workflows, retrieves the most relevant verified guideline passages, and assembles the evidence shown to the dermatologist. When critical information is missing or a tool fails, it routes the case to manual review. It never sets, modifies or overrides the validated risk score or the deterministic ranking that fills the daily slots.**
- **Tools:** `check_image_quality`, `request_retake`, `ask_patient`, `predict_visual_risk`, `score_symptoms`, `retrieve_guidance_context`, `draft_biopsy_referral`.
- On failure: Fall back to the non-agent path. Any tool error, low confidence or missing critical input routes the case to a separate manual-review queue and never lowers its validated model priority or removes its regular appointment.
- **Guardrails:**
  * **The agent never issues a biopsy referral and never books an appointment on its own. It only drafts, and the dermatologist approves every action.**
  * Strict prohibition of "benign" or "safe" diagnostic wording in the dashboard or patient communications.
  * The patient receives a neutral message ("your photo was received and will be reviewed") and never a risk estimate.
  * Explicit "I don't know" fallback if clinical inquiries fall outside the verified NCI PDQ guidance corpus.
  * Patient-submitted text is treated as data, never as instructions.
  * Users under the age of 18 are automatically routed to direct clinical review regardless of AI score.

## 6 · Risks and cut line
| Risk | Mitigation |
|---|---|
| **Missing fields leak the label** (In PAD-UFES-20, missing field counts alone yield an artifactual AUC of 0.83) | Explicitly restrict training to the subset of features uniformly collected by the app's mandatory onboarding flow. |
| **Patient / lesion leakage** (Naive splits may place different photos of the same patient or lesion across train and test; HAM10000's 10,015 images represent only 7,470 distinct lesions, and MILK10k does not expose a patient ID) | Split at the **Patient ID** level wherever one exists (PAD-UFES-20), and at the lesion level where it does not (MILK10k, HAM10000). Splits are generated once, frozen, and reused by every experiment. |
| **Prevalence mismatch** (about 57% of the clinical images in the training pool are malignant because these datasets over-represent lesions selected for biopsy; real-world prevalence is much lower) | Use capacity-based ranking rather than a fixed probability threshold, and evaluate the system at 5%, 10% and 20% malignant prevalence. Report sensitivity / recall@k at fixed review capacity. Prevalence shift remains an explicit limitation. |
| **Data domain gap** (patients may submit photos with poor lighting, blur, framing problems or lens artefacts) | Build an image-quality dataset by manually labelling ~300 PAD images against a written usable/unusable rubric. Split these labels at the patient level; apply synthetic degradations (blur, under-/over-exposure, colour cast) only to the training portion, and evaluate the quality filter on held-out manually labelled patients. |
| **No verified dark-skin test data.** Both skin-tone sources skew light | Report M2 per Fitzpatrick bin with confidence intervals; state that Fitzpatrick V–VI performance is unvalidated, and list it as the first requirement for a clinical pilot |
| **Melanoma-specific statistics are thin.** PAD-UFES-20 contains only 52 melanomas in total, so the held-out test split will contain a relatively small number of melanoma patients. | Report melanoma sensitivity with 95% confidence intervals and raw TP/FN counts. Treat the headline M2 target as applying to skin cancer overall (MEL + BCC + SCC), while melanoma performance is reported separately and interpreted cautiously. |

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
| Data | Tami Khazma | Pipeline architecture, strict patient-level splitting, data cleaning, and leakage mitigation |
| Model | Roi Budnitsky | Training baselines, model fine-tuning, threshold tuning, and multi-ethnic performance analysis |
| Agent | Yuval Rubenchuk | Agent tool building, orchestration loops, safety guardrails, and M4 validation set execution |
| Product | Ifat Davidson | Clinician workflow UX, patient-facing safety copy, README documentation, and final fund presentation deck |

---
¹ Murad H et al. *Measuring geographical disparities in waiting times for community-based specialist care.* Israel Journal of Health Policy Research, 2025. [doi:10.1186/s13584-025-00702-7](https://doi.org/10.1186/s13584-025-00702-7). Covers all dermatologists in all four health funds, 2019–2023.

² Maccabi, [online dermatologist consultation](https://www.maccabi4u.co.il/31276/digital-services/communication/consultation_dermatologists/): "not intended for urgent cases or for diagnosing moles and skin lesions". Clalit, [online dermatologists](https://www.clalit.co.il/he/online_doctors/Pages/dermatologist_on_line.aspx): excludes "diagnosis and assessment of nevi" and suspected skin cancer. Both checked 18.9.2026.
