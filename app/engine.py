"""The demo app's logic, kept free of Streamlit so it can be tested on its own.

Everything here calls the project's real code where it exists:
  - questionnaire score  -> src.questionnaire (models/questionnaire.joblib)
  - image score          -> src.predict.VisualRiskModel (downloads the weights on first use; CPU works)
  - ranking              -> src.fusion.fuse and rank_band, against 99 real validation patients
  - agent                -> agent/triage_agent.py: code makes the decision (decide); the live CrewAI agent
                            reports on the note and writes the texts, and without a key fixed texts are used.
                            In both cases the real enforce_guardrails and Word report
What is NOT the project's: the photo-quality check below is a simple brightness/sharpness heuristic,
because the project has no quality classifier yet.
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "agent"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
os.environ.setdefault("RAG_DIR", str(ROOT / "data" / "processed" / "rag"))

import triage_agent as T  # noqa: E402
from src.fusion import fuse, rank_band  # noqa: E402

SYMPTOMS = ("itch", "grew", "hurt", "changed", "bleed", "elevation")
SITES = {"FACE": "פנים", "NOSE": "אף", "EAR": "אוזן", "LIP": "שפה", "SCALP": "קרקפת", "NECK": "צוואר",
         "CHEST": "חזה", "BACK": "גב", "ABDOMEN": "בטן", "ARM": "זרוע", "FOREARM": "אמה", "HAND": "כף יד",
         "THIGH": "ירך", "FOOT": "כף רגל"}
QUEUE_SIZE, SLOTS, W_IMAGE = 100, 10, 0.6


# ── 1. photo quality (heuristic, see module docstring) ────────────────────────────────────
def photo_quality(img: Image.Image) -> dict:
    im = img.convert("RGB")
    s = 256 / max(im.size)
    im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))))
    a = np.asarray(im, dtype=float)
    L = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    lap = (-4 * L[1:-1, 1:-1] + L[:-2, 1:-1] + L[2:, 1:-1] + L[1:-1, :-2] + L[1:-1, 2:]).var()
    mean, bright = float(L.mean()), float((L > 245).mean())
    issues = []
    if mean < 80:
        issues.append("too dark")
    if mean > 215 or bright > 0.25:
        issues.append("too bright")
    elif bright > 0.04:
        issues.append("glare")
    if mean >= 80 and lap < 6:
        issues.append("blurred")
    status = "usable"
    if mean < 60 or (mean >= 80 and lap < 2):
        status = "unusable"
    elif issues:
        status = "borderline"
    return {"status": status, "issues": issues, "brightness": round(mean), "sharpness": round(float(lap), 1)}


# ── 2. questionnaire model: the project's saved pipeline ─────────────────────────────────
@lru_cache(maxsize=1)
def questionnaire_model():
    from src.questionnaire import load_model
    return load_model()


def questionnaire_score(age, site, answers: dict) -> float:
    from src.questionnaire import score
    row = {"age": age, "region": site or np.nan}
    row.update({k: {"yes": "TRUE", "no": "FALSE"}.get(answers.get(k), np.nan) for k in SYMPTOMS})
    return float(score(questionnaire_model(), pd.DataFrame([row]))[0])


# ── 3. image model: the project's VisualRiskModel, when torch and the weights are available ──
@lru_cache(maxsize=1)
def image_model():
    """None when torch/timm are missing or the weights cannot be fetched; the app then asks for a score."""
    try:
        from src.predict import VisualRiskModel
        return VisualRiskModel()            # GPU when Colab gives one, else CPU
    except Exception as e:  # noqa: BLE001
        print(f"image model unavailable: {type(e).__name__}: {e}")
        return None


def image_score(photo_path: str) -> float | None:
    m = image_model()
    return None if m is None else float(m.predict([photo_path])["cancer_score"])


# ── 4. today's queue: 99 real validation patients + the new one ──────────────────────────
@lru_cache(maxsize=1)
def today_queue(seed: int = 7) -> pd.DataFrame:
    v = pd.read_csv(ROOT / "data" / "processed" / "val_scores_fused.csv")
    rng = np.random.default_rng(seed)
    mal = v[v.y == 1].sample(10, random_state=int(rng.integers(1e9)))
    ben = v[v.y == 0].sample(QUEUE_SIZE - 11, random_state=int(rng.integers(1e9)))
    return pd.concat([mal, ben])[["patient_id", "image_score", "questionnaire_score"]].reset_index(drop=True)


def rank_today(img: float, quest: float, new_id: str = "NEW") -> dict:
    q = pd.concat([today_queue(), pd.DataFrame([{"patient_id": new_id, "image_score": img,
                                                 "questionnaire_score": quest}])]).set_index("patient_id")
    q["fused"] = fuse(q["image_score"], q["questionnaire_score"], W_IMAGE)          # the project's fusion
    q["image_pct"] = q["image_score"].rank(pct=True)
    q["quest_pct"] = q["questionnaire_score"].rank(pct=True)
    q = q.reset_index().sort_values(["fused", "patient_id"], ascending=[False, True]).reset_index(drop=True)
    q["rank"] = q.index + 1
    q["band"] = q["fused"].map(rank_band)
    me = q[q.patient_id == new_id].iloc[0]
    return {"queue": q, "rank": int(me["rank"]), "fused": float(me["fused"]), "band": me["band"],
            "image_band": rank_band(me["image_pct"]), "symptom_band": rank_band(me["quest_pct"]),
            "flagged": int(me["rank"]) <= SLOTS}


# ── 5. the agent ──────────────────────────────────────────────────────────────────────────
def make_case(pid, rank, r, age, site, answers, note, photo_path, quality) -> dict:
    """A case in exactly the shape agent/triage_agent.py reads from daily_cases.csv."""
    return {"patient_id": pid, "rank": rank, "risk_score": round(r["fused"], 4), "risk_band": r["band"],
            "image_risk_band": r["image_band"], "symptom_risk_band": r["symptom_band"],
            "age": age, "body_site": (site or "").lower(),
            "symptoms": {k: answers.get(k) or "unknown" for k in SYMPTOMS},
            "patient_note": note or "", "image_path": photo_path,
            "image_quality": quality["status"], "image_issues": list(quality["issues"])}


def no_llm_report(case: dict) -> "T.AgentReport":
    """Stands in for the agent when there is no LLM key. Code still makes the decision (T.decide);
    the only thing it reports is that the patient wrote a note, so the note reaches the dermatologist flagged."""
    note = (case.get("patient_note") or "").strip()
    return T.AgentReport(patient_id=case["patient_id"],
                         note_findings=[T.NoteFinding(kind="new_information", quote=note)] if len(note) >= 3 else [])


def live_agent_ready() -> tuple[bool, str]:
    """Can the live CrewAI agent run here? (an API key and crewai installed)"""
    import importlib.util
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        return False, "no GEMINI_API_KEY"
    if importlib.util.find_spec("crewai") is None:
        return False, "crewai is not installed"
    return True, os.getenv("TRIAGE_MODEL", "gemini-3.5-flash-lite")


def review(case: dict, mode: str) -> tuple["T.CaseReview", str]:
    """mode: 'off' (no agent), 'rules' (decision rules, no LLM) or 'live' (CrewAI + the configured LLM)."""
    T.CASES = {case["patient_id"]: case}
    if mode == "off":
        return T.no_agent_review(case), "agent off: fixed model-evidence summary"
    if mode == "live":
        try:
            crew = T.build_crew()
            return T.review_case(crew, case), "live agent"
        except Exception as e:  # noqa: BLE001  no key, no crewai: say so and fall back
            note = f"live agent unavailable ({type(e).__name__}); decision rules used instead"
    else:
        note = "decision rules (no LLM)"
    return T.enforce_guardrails(no_llm_report(case), case, check_quotes=False), note


def word_report(case: dict, rv, out_path: str) -> str:
    import datetime as dt
    return T.build_word_report({case["patient_id"]: case}, {case["patient_id"]: rv}, out_path, dt.date.today(),
                               agent_used=rv.decision != "model_evidence_only")
