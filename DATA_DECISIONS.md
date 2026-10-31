# Data Decisions · Skin Lesion Triage for Healthcare Providers


This file backs the data section of [CHARTER.md](CHARTER.md). Initial source checks were performed on 18–19 September 2026 (DDI checked on 22 September). The implementation details and results below were updated through 31 October 2026. Historical source-screening findings are distinguished from the final modelling datasets and evaluations.



## 1 · Four checks per source

| Source | Access | Size | Licence | Signal |
|---|---|---|---|---|
| **ISIC Archive — clinical close-ups, including MILK10k** | ISIC Archive API and image downloads; source metadata recorded in `data/raw/isic_clinical_metadata.csv` | 9,316 clinical images initially identified. Excluding the PAD copy (2,298) and unlabelled benchmark (479) leaves 6,539 candidates. After preprocessing exclusions, **6,404 labelled images** remain (3,687 malignant, 470 melanomas). | CC-BY-NC, CC-BY and CC-0, depending on the image | Main source for image-model training. The ISIC-only model achieved patient AUC 0.778 on PAD validation, compared with 0.956 after PAD fine-tuning. This demonstrates the importance of adapting to the target domain. |
| **PAD-UFES-20** | Official metadata from Mendeley; photographs from a public mirror verified against PAD images in ISIC | **2,298 images, 1,891 distinct lesions, 1,373 patients, 52 melanomas; 47.4% malignant** | CC BY 4.0 | Target domain of smartphone photographs and the source of symptom answers. Used for patient-level training, validation and held-out testing. The frozen questionnaire baseline achieved patient AUC 0.867 on validation. |
| **NCI PDQ — Health Professional summaries** | NCI Cancer.gov; frozen text stored in `data/processed/rag/nci_corpus.json` | **7 documents, 224 sections, 724 chunks** | NCI public-domain text; credit NCI | Official clinical guidance corpus for RAG. Citation-ID groundedness improved from 0.733 to 0.933 after re-ranking. The 5/5 out-of-corpus refusals were tested before re-ranking only. |

**ISIC licensing note.** The initial 9,316-image screening pool contained 5,719 CC-BY-NC, 3,545 CC-BY and 52 CC-0 images. These counts describe the initial source pool, not the final 6,404-image dataset. Commercial reuse requires verification of the licences applicable to the actual images and trained model.

**Dataset scope.** The 6,404 ISIC images form the cleaned labelled pool; not all were used for gradient-based training. The training and evaluation subsets are documented in `data/data_dictionary.md`, `src/README.md` and `reports/results_image_model.md`.


## 2 · Rejected sources
| Source | Reason |
|---|---|
| **HAM10000 on Hugging Face** (`marmal88/skin_cancer`) | Its ready-made split leaks. The 13,354 rows contain only 10,015 unique images; **79.8% of the test images are also in train**, and 88.1% share a lesion with train. We use the original source instead. |
| **HAM10000** (original source, considered for pretraining) | Rejected — `reports/decisions.md` #8. It overlaps ISIC, so pretraining on it risks the same lesion appearing in training and in evaluation, and it is dermoscopic: 10,015 images and 1,113 melanomas, but none of them a phone photo, so it cannot narrow the domain gap that actually matters here. |
| **BCN20000** | Dermoscopic only. Beyond HAM10000 it adds volume and no new domain, so it cannot narrow the gap to a phone photo. |
| **Fitzpatrick 17k** | Only **10 of 40 sampled image links still work**. Atlas photos rather than phone photos, mostly non-cancerous conditions, and no age, sex or body site. Its one advantage, diverse skin tones, does not survive the dead links. |
| **DDI** (Diverse Dermatology Images, Stanford) · checked 22.9 | The only biopsy-proven source with balanced dark skin tones: 656 images, 570 patients, 171 malignant, 78 diagnoses, split 208 / 241 / 207 across Fitzpatrick I–II, III–IV and V–VI. Rejected on three grounds. **Access** requires a signed Stanford Research Use Agreement, which may not arrive before the deadline. **Licence** is non-commercial and forbids redistribution, so nothing derived from it may enter a public repo. **Domain**: the images are clinic photographs taken at Stanford between 2010 and 2020, not smartphone photos, so a gap measured on it would confound skin tone with image domain. Skin tone is reported from the PAD Fitzpatrick field and the MILK10k field instead. |
| **Generative "phone-to-dermoscopic" conversion** | A dermatoscope (≈10× magnification, polarised light) captures subsurface structures that a phone camera cannot. A generator would invent them, which is unsafe for a medical app. |

## 3 · Findings that shape the design
- **Missing fields leak the label.** In PAD-UFES-20, optional fields such as medical history, smoking and Fitzpatrick skin type are recorded unevenly, largely because the dataset includes different clinical and biopsy workflows. The count of missing optional fields alone achieves **ROC-AUC 0.839** on development data, revealing an administrative shortcut rather than a clinically meaningful signal. The final questionnaire baseline therefore uses only age, body site and the six symptom questions. Missing answers are imputed inside the training pipeline, without missingness indicators, and `biopsed` is excluded entirely. The frozen questionnaire baseline achieves patient AUC **0.867** on validation (`reports/results.md` §6).
- **Symptoms separate cancer types.** "Bleeds" and "hurts" point to BCC and SCC (0% melanoma). **"Changed"** is the strongest melanoma signal: 16.8% melanoma when yes vs 0.5% when no.
- **Leakage through duplicates.** A naive image-level split puts 43% of PAD test images next to their own lesion in train by raw lesion id, and **29.3%** once lesion ids are namespaced by patient, which is the figure the project reports (39% for HAM10000). MILK10k has no patient IDs. PAD reuses **250 of its 1,641 lesion ids across different patients**, so the raw id is not a lesion and the key must be `patient_id::lesion_id` — **1,891 distinct lesions**. PAD also exists twice, on Mendeley and inside ISIC; we use one copy only.
- **Prevalence mismatch.** The cleaned ISIC pool contains **57.6% malignant images**, and PAD-UFES-20 contains **47.4% malignant images**. Both datasets substantially overrepresent suspicious or biopsy-selected lesions compared with a general dermatology waiting list. The system therefore uses a capacity-based ranking policy rather than a fixed malignancy-probability threshold. The product policy flags the highest-risk **10% of the current waiting list**, while the primary M2 evaluation uses a standardized scenario of **10 priority-review slots per 100 arrivals at an assumed 10% malignancy prevalence**. These are distinct definitions. Capacity-based ranking does not eliminate selection bias or guarantee performance under real-world prevalence shifts.


## 4 · Critical-inquiry experiments, and what became of them

| # | Experiment, as planned on 19.9 | Outcome |
|---|---|---|
| 1 | **Domain gap:** train without PAD, test only on PAD (clinic photos vs. real phone photos). | **Done.** ConvNeXt trained on ISIC alone reaches patient AUC 0.778 against 0.956 after PAD fine-tuning, and most of the gap is actinic keratosis: without the 81 patients who have an AK the ISIC-only model already reaches 0.899. `reports/results.md` §7. |
| 2 | **Skin tone:** M2 per Fitzpatrick bin, from the PAD Fitzpatrick field and the MILK10k skin-tone field. | **Abandoned, and stated as a limitation instead.** PAD holds 11 Fitzpatrick V–VI images in total and ISIC's 376 all come from one contributor inside the training pool, so any measured gap would be the institution, not the skin tone. Reported as raw counts; a dark-skin evaluation set is the first requirement for a pilot. |
| 3 | **Age shortcut:** M2 for patients under 40, reported separately. | **Not run.** Age carries real signal (odds ratio 1.28 per ten years) and "young = benign" remains an unmeasured risk. |
| 4 | **Overlap check:** MILK10k lesions that also appear in HAM10000. | **Moot** — HAM10000 was rejected, so nothing is pretrained on it. |

## 5 · Licensing and the path to a real product
Most images are CC-BY-NC, which is fine for the course. A commercial pilot would need:
- data licences from the contributing institutions, **or** our own data collection with Helsinki-committee approval and informed consent;
- regulatory approval as a medical device (in Israel from the MoH medical-device division, AMAR; CE in Europe; FDA in the US);
- legal advice on whether a model trained on non-commercial data may be used commercially.

