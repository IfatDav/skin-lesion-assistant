# Data Decisions · Skin Lesion Triage for Healthcare Providers

This file backs the data section of [CHARTER.md](CHARTER.md). All numbers come from checks against the original sources, run on 18–19.9.2026 (DDI checked 22.9.2026).

## 1 · Four checks per source
| Source | Access | Size | Licence | Signal |
|---|---|---|---|---|
8,837 labelled images (9,316 clinical close-ups minus the 479 unlabelled benchmark; still includes PAD's ISIC copy): **522 melanomas**, 1,005 nevi.. MILK10k: 5,240 lesions, 95.7% confirmed by histopathology, skin-tone field 0–5 |  Per image: CC-BY-NC 5,719 (all of MILK10k) · CC-BY 3,545 · CC-0 52. Only 72 melanomas are usable commercially | Age + sex + site alone: ROC-AUC 0.74 (GroupKFold) |
| **PAD-UFES-20** (test set of real phone photos; images from patients **outside** the test split also train the symptom model) | Metadata CSV from Mendeley; images also in ISIC | 2,298 images · 1,641 raw lesion ids = **1,891 distinct lesions** · 1,373 patients · **52 melanomas** | CC BY 4.0 (checked on Mendeley) | Age + site + 6 symptoms: ROC-AUC 0.90 (GroupKFold by patient) — measured before the missingness policy was frozen; the honest baseline is variant D, CV AUC 0.860 and validation patient AUC 0.867 (`reports/results.md` §6) |
| **NCI PDQ** patient summaries (RAG corpus) | cancer.gov | ~10–20 documents | "All text within NCI products is free of copyright"; credit NCI | Coverage checked while writing the M3 question set. The module 5 retrieval work showed why a dedicated corpus is needed: a question about official treatment guidelines correctly returned "I don't know", because arXiv abstracts do not contain them |

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
- **Missing fields leak the label.** In PAD, fields such as smoking, history and Fitzpatrick type are missing in 60–85% of benign cases and 0% of cancers. The *count of missing fields alone* reaches ROC-AUC 0.83, and "all fields" reaches 0.949, partly through this leak. We use only fields the app always asks.
- **Symptoms separate cancer types.** "Bleeds" and "hurts" point to BCC and SCC (0% melanoma). **"Changed"** is the strongest melanoma signal: 16.8% melanoma when yes vs 0.5% when no.
- **Leakage through duplicates.** A naive image-level split puts 43% of PAD test images next to their own lesion in train by raw lesion id, and **29.3%** once lesion ids are namespaced by patient, which is the figure the project reports (39% for HAM10000). MILK10k has no patient IDs. PAD reuses **250 of its 1,641 lesion ids across different patients**, so the raw id is not a lesion and the key must be `patient_id::lesion_id` — **1,891 distinct lesions**. PAD also exists twice, on Mendeley and inside ISIC; we use one copy only.
- **Prevalence mismatch.** 51% of the ISIC clinical images are malignant, because they are lesions that were chosen for biopsy. The product answers this by design: it fills a fixed daily capacity (top-10) instead of applying a probability threshold, so a miscalibrated score shifts every number equally and leaves the ranking intact.

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

