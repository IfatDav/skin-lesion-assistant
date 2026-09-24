# Data Decisions · Skin Lesion Triage for Healthcare Providers

This file backs the data section of [CHARTER.md](../CHARTER.md). All numbers come from checks against the original sources, run on 18–19.9.2026 (DDI checked 22.9.2026).

## 1 · Four checks per source
| Source | Access | Size | Licence | Signal |
|---|---|---|---|---|
| **ISIC Archive**, clinical close-ups, incl. MILK10k (main training data) | ✅ Public API, no account | ✅ 8,837 labelled images (479 without a diagnosis dropped): **522 melanomas**, 1,005 nevi. MILK10k: 5,240 lesions, 95.7% confirmed by histopathology, skin-tone field 0–5 | ⚠️ Per image: CC-BY-NC 5,719 (all of MILK10k) · CC-BY 3,545 · CC-0 52. Only 72 melanomas are usable commercially | ✅ Age + sex + site alone: ROC-AUC 0.74 (GroupKFold) |
| **PAD-UFES-20** (test set of real phone photos; images from patients **outside** the test split also train the symptom model) | ✅ Metadata CSV from Mendeley; images also in ISIC | ⚠️ 2,298 images · 1,641 lesions · 1,373 patients · **52 melanomas** | ✅ CC BY 4.0 (checked on Mendeley) | ✅ Age + site + 6 symptoms: ROC-AUC 0.90 (GroupKFold by patient) |
| **HAM10000** (pretraining only) | ✅ Harvard Dataverse, direct download | ✅ 10,015 images · 7,470 lesions · 1,113 melanomas | ⚠️ CC BY-NC 4.0 (checked on Dataverse) | ✅ Established for dermoscopic images; limited for phone photos (different domain) |
| **NCI PDQ** patient summaries (RAG corpus) | ✅ cancer.gov | ✅ ~10–20 documents | ✅ "All text within NCI products is free of copyright"; credit NCI | ⚠️ Coverage checked while writing the M3 question set. The module 5 retrieval work showed why a dedicated corpus is needed: a question about official treatment guidelines correctly returned "I don't know", because arXiv abstracts do not contain them |

## 2 · Rejected sources
| Source | Reason |
|---|---|
| **HAM10000 on Hugging Face** (`marmal88/skin_cancer`) | Its ready-made split leaks. The 13,354 rows contain only 10,015 unique images; **79.8% of the test images are also in train**, and 88.1% share a lesion with train. We use the original source instead. |
| **BCN20000** | Dermoscopic only. Beyond HAM10000 it adds volume and no new domain, so it cannot narrow the gap to a phone photo. |
| **Fitzpatrick 17k** | Only **10 of 40 sampled image links still work**. Atlas photos rather than phone photos, mostly non-cancerous conditions, and no age, sex or body site. Its one advantage, diverse skin tones, does not survive the dead links. |
| **DDI** (Diverse Dermatology Images, Stanford) · checked 22.9 | The only biopsy-proven source with balanced dark skin tones: 656 images, 570 patients, 171 malignant, 78 diagnoses, split 208 / 241 / 207 across Fitzpatrick I–II, III–IV and V–VI. Rejected on three grounds. **Access** requires a signed Stanford Research Use Agreement, which may not arrive before the deadline. **Licence** is non-commercial and forbids redistribution, so nothing derived from it may enter a public repo. **Domain**: the images are clinic photographs taken at Stanford between 2010 and 2020, not smartphone photos, so a gap measured on it would confound skin tone with image domain. Skin tone is reported from the PAD Fitzpatrick field and the MILK10k field instead. |
| **Generative "phone-to-dermoscopic" conversion** | A dermatoscope (≈10× magnification, polarised light) captures subsurface structures that a phone camera cannot. A generator would invent them, which is unsafe for a medical app. |

## 3 · Findings that shape the design
- **Missing fields leak the label.** In PAD, fields such as smoking, history and Fitzpatrick type are missing in 60–85% of benign cases and 0% of cancers. The *count of missing fields alone* reaches ROC-AUC 0.83, and "all fields" reaches 0.949, partly through this leak. We use only fields the app always asks.
- **Symptoms separate cancer types.** "Bleeds" and "hurts" point to BCC and SCC (0% melanoma). **"Changed"** is the strongest melanoma signal: 16.8% melanoma when yes vs 0.5% when no.
- **Leakage through duplicates.** A naive image-level split puts 43% of PAD test images next to their own lesion in train (39% for HAM10000). MILK10k has no patient IDs. PAD lesion IDs differ between Mendeley (1,641) and ISIC (1,891). PAD also exists twice, on Mendeley and inside ISIC; we use one copy only.
- **Prevalence mismatch.** 51% of the ISIC clinical images are malignant, because they are lesions that were chosen for biopsy. The product answers this by design: it fills a fixed daily capacity (top-10) instead of applying a probability threshold, so a miscalibrated score shifts every number equally and leaves the ranking intact.

## 4 · Critical-inquiry experiments (planned)
1. **Domain gap:** train without PAD, test only on PAD (camera photos vs. real phone photos).
2. **Skin tone:** M2 (sensitivity) per Fitzpatrick bin, on two sources: the Fitzpatrick field in PAD-UFES-20, our test set, and the MILK10k skin-tone field (0–5) inside ISIC. Both skew light, so performance on Fitzpatrick V–VI stays an open limitation that we state rather than hide.
3. **Age shortcut:** M2 for patients under 40, reported separately, because age is the strongest metadata signal and "young = benign" is the shortcut we most fear.
4. **Overlap check:** MILK10k lesions that also appear in HAM10000 (via the ISIC ID in the MILK10k metadata) are removed before pretraining and evaluation.

## 5 · Licensing and the path to a real product
Most images are CC-BY-NC, which is fine for the course. A commercial pilot would need:
- data licences from the contributing institutions, **or** our own data collection with Helsinki-committee approval and informed consent;
- regulatory approval as a medical device (in Israel from the MoH medical-device division, AMAR; CE in Europe; FDA in the US);
- legal advice on whether a model trained on non-commercial data may be used commercially.


```mermaid
flowchart LR
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
  
