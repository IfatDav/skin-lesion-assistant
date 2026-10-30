# Ink labelling · surgical-marker shortcut check

**Why.** Purple surgical-marker ink was seen next to a lesion on a PAD photo. Doctors mark lesions before a
biopsy, and every cancer in PAD was biopsied, so a model can learn "ink = cancer" instead of looking at the
lesion. This is a known failure of dermatology models (Winkler et al., *JAMA Dermatology* 2019). An automatic
colour detector was tried and failed (it missed the photo where ink was seen), so the check needs human eyes.

**The sheet.** `ink_labeling_test.csv` is written by the single test run (`python -m src.evaluate_test
--frozen`): every PAD test photo, in random order, with no diagnosis and no score, so the labeller is not
influenced by them. Labelling happens after the test run, as decided in `reports/decisions.md`; it does not
change the model, so it does not touch the test result.

## How to label (about 500 photos, under an hour)

1. Open each photo from the `photo` column. The paths are relative to the repo root and point *outside*
   it, to the `DATA/` folder described in `src/README.md` — set `SKIN_DATA_ROOT` or place `DATA/` beside
   the repo. Alternatively run `python -m src.build_index --resize-pad` to cache 512-px copies inside
   `data/processed/pad_512/`.
2. Fill `has_ink` with **1** if there is any marker ink on the skin: lines, dots, a circle around the lesion,
   arrows, written letters. Purple and blue are the usual colours.
3. Fill **0** if there is none. Natural pigment, bruises, veins, hair and shadows are not ink.
4. If unsure, write **1** and describe it in `notes`.

## Then

```bash
python -m src.ink_effect
```

It reports how often ink appears in cancer vs non-cancer photos, and whether the image model scores inked
photos higher **within the same diagnosis** — the actual sign of a shortcut.
