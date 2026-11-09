# Demo app (Streamlit)

A patient sends a photo and six answers; the app shows what happens next, using the project's own code.

| Step | What runs | Real or demo |
|---|---|---|
| Photo quality | brightness and sharpness check in `engine.py` | demo heuristic (the project has no quality classifier yet) |
| Questionnaire score | `models/questionnaire.joblib` via `src.questionnaire` | real |
| Image score | `src.predict.VisualRiskModel` (ConvNeXt, CPU) when the switch is on; otherwise a slider | real when switched on |
| Today's queue | `src.fusion.fuse` + `rank_band` against 99 real validation patients | real |
| Agent | live CrewAI agent (needs `GEMINI_API_KEY`), or the agent's decision rules in code | live or rules; both pass the real `enforce_guardrails` |
| Word report | `build_word_report` from `agent/triage_agent.py` | real |

## Run on your computer

From the repository root:

```bash
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```

The browser opens at http://localhost:8501. The first time the image-model switch is turned on, the weights
(about 110 MB) are downloaded from the project's GitHub release.

## Put it online (Streamlit Community Cloud, free)

1. Sign in at share.streamlit.io with GitHub.
2. **Create app** → choose this repository, branch `main`, main file `app/streamlit_app.py`.
3. Optional, for the live agent: **Advanced settings → Secrets** → `GEMINI_API_KEY = "..."`. Never commit the key.
4. **Deploy.** The first build takes a few minutes (it installs PyTorch). The app gets a public address
   to put in the README and the presentation.

Example photos in `examples/`: PAD-UFES-20 (Pacheco et al., CC BY 4.0); `dark.jpg` is darkened on purpose
to show the retake path. The example patients' ages and answers are made up.
