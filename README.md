# Project Charter · Skin Lesion Triage for Healthcare Providers

**Team:** Ifat Davidson · Tami Khazma · Roi Budnitsky · Yuval Rubenchuk | **Course:** BIU DS23 Capstone | **Date:** 25.9.2026
**Retrieval asset:** the module 5 grounded assistant.

## 1 · Problem & User
**User:** A dermatologist working within an Israeli healthcare fund (e.g., Clalit, Maccabi) who manages an overloaded appointment queue.

**Pain:**  the dermatologist cannot tell which patient on the waiting list cannot wait.
A public-system dermatologist appointment in Israel takes a median of **24.5 days**, rising by **8.5 days a year**, and reaches **40–42 days** in Tel Aviv and Jerusalem ¹.
Clalit's and Maccabi's online dermatology services explicitly exclude moles and suspected skin cancer.² These cases therefore require in-person assessment. In the absence of a dedicated pre-visit image-triage mechanism, the waiting list itself does not reveal which lesions are most urgent. A malignant lesion may therefore remain in the standard queue until it is reviewed by a clinician.

**What we build:** while waiting for their appointment, The patient uploads a smartphone photograph of the lesion and fills out a brief symptom form via the healthcare fund's app. The system scores every patient on the waiting list. Every morning it shows the dermatologist a **short list of up to 10 suspicious cases**, each with the photo, the answers and cited reasons.
For each case, the dermatologist decides whether to move the patient to an earlier in-person appointment and, where clinically and operationally appropriate, whether to issue a biopsy referral in advance to accelerate the diagnostic workup.

**Safety principle:** the system can only move a patient *earlier*. A patient who is not flagged keeps their regular appointment, so a miss costs at most today's situation. The patient never sees a risk score or a diagnosis, and the dermatologist decides on every case.
  
**AI-deletion test:** Without AI the problem remains: dozens of photos arrive every day, and no dermatologist can review them all in time to find the few that cannot wait..


##  Architecture

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
It works end to end without the agent: both models score each patient, the deterministic ranking policy fills the day's urgent slots, and each case receives a fixed model-evidence summary containing the relevant score bands, symptom flags and image-quality status
