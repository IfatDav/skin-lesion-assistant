"""The demo app's logic, kept free of Streamlit so it can be tested on its own.

Everything here calls the project's real code where it exists:
  - questionnaire score  -> src.questionnaire (models/questionnaire.joblib)
  - image score          -> src.predict.VisualRiskModel (downloads the weights on first use; CPU works)
  - ranking              -> src.fusion.fuse and rank_band, against 99 real validation patients
  - agent                -> agent/triage_agent.py: the live CrewAI agent, or the decision rules,
                            and in both cases the real enforce_guardrails and Word report
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
LEVEL = {"low": 0, "moderate": 1, "high": 2, "very high": 3}
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


# Hebrew final letters change form (החשיך / החשיכה), so the stems are listed without them.
NOTE_WORDS = {"bleed": ["מדמם", "דימום", "דמם", "bleed"], "grew": ["גדל", "התרחב", "grew", "bigger"],
              "changed": ["השתנ", "החשיך", "החשיכ", "changed", "darker"], "itch": ["מגרד", "itch"],
              "hurt": ["כואב", "כאב", "hurt", "pain"], "elevation": ["בולט", "raised"]}
ASK = {"bleed": "כתבת שהנגע מדמם, ובטופס סימנת שלא. האם הוא מדמם גם בלי שנוגעים בו?",
       "grew": "כתבת שהנגע גדל, ובטופס סימנת שלא. בערך מתי שמת לב לזה?",
       "changed": "כתבת שהנגע השתנה, ובטופס סימנת שלא. האם השתנה הצבע, הצורה או הגודל?",
       "itch": "כתבת שהנגע מגרד, ובטופס סימנת שלא. האם זה קורה כל יום?",
       "hurt": "כתבת שהנגע כואב, ובטופס סימנת שלא. האם הכאב מופיע גם בלי מגע?",
       "elevation": "כתבת שהנגע בולט, ובטופס סימנת שלא. האם הוא היה בולט גם בעבר?"}


def rules_review(case: dict) -> "T.CaseReview":
    """The agent's DECISION_RULES applied by code, for running without an LLM key.
    The result still goes through the project's own enforce_guardrails."""
    pid, q = case["patient_id"], case["image_quality"]
    why_on = T.why_on_list(case)
    base = dict(patient_id=pid, why_on_list=why_on, evidence_summary=T.model_evidence(case))
    if case["age"] is None or case["age"] < 18:
        return T.CaseReview(decision="manual_review", decision_reason="Under 18 or age missing.", **base)
    if q in ("unusable", "borderline"):
        if q == "borderline" and case["risk_band"] == "very high":
            return T.CaseReview(decision="proceed_with_flag",
                                decision_reason="Borderline photo, but very high risk: not delayed.", **base)
        fix = T.retake_needed(case)
        if fix:
            return T.CaseReview(decision="request_retake", decision_reason=f"Photo issue a retake can fix: {', '.join(fix)}.",
                                retake_instruction=" ".join(T.RETAKE_FIX_HE[k] for k in fix[:2]), **base)
        return T.CaseReview(decision="proceed_with_flag", decision_reason="Borderline photo; a retake would not help.", **base)
    note = case["patient_note"].lower()
    for k, words in NOTE_WORDS.items():
        if case["symptoms"][k] == "no" and any(w in note for w in words):
            return T.CaseReview(decision="ask_patient", decision_reason=f"The note mentions '{k}', the form says no.",
                                follow_up_question=ASK[k], **base)
    if abs(LEVEL[case["image_risk_band"]] - LEVEL[case["symptom_risk_band"]]) >= 2:
        return T.CaseReview(decision="ask_patient", decision_reason="Image and symptom bands are two or more levels apart.",
                            follow_up_question="האם התמונה מראה את הנגע שמדאיג אותך? אם לא, צלמו אותו שוב.", **base)
    if note.strip():
        return T.CaseReview(decision="proceed_with_flag", decision_reason="The note adds information the form does not capture.", **base)
    return T.CaseReview(decision="proceed", decision_reason="Usable photo, consistent information, nothing missing.", **base)


def note_contradiction(case: dict) -> str | None:
    """The first symptom the patient's note mentions while the form says 'no' (checked in code)."""
    note = (case.get("patient_note") or "").lower()
    for k, words in NOTE_WORDS.items():
        if case["symptoms"].get(k) == "no" and any(w in note for w in words):
            return k
    return None


def note_check(rv: "T.CaseReview", case: dict) -> "T.CaseReview":
    """Guardrail: a note that contradicts the form must reach the patient as a question.
    Rule 3 of the agent's own rules; the LLM can miss it (e.g. a Hebrew note), so code enforces it.
    Decisions that rank above ask_patient (manual_review, request_retake) are left alone."""
    k = note_contradiction(case)
    if k is None or rv.decision not in ("proceed", "proceed_with_flag"):
        return rv
    return rv.model_copy(update={
        "decision": "ask_patient",
        "decision_reason": f"The note mentions '{k}', the form says no (corrected by code; the agent chose {rv.decision}).",
        "follow_up_question": ASK[k],
        "flags": list(rv.flags) + [f"note contradicts form ({k}): decision corrected by code"]})


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
            return note_check(T.review_case(crew, case), case), "live agent"
        except Exception as e:  # noqa: BLE001  no key, no crewai: say so and fall back
            note = f"live agent unavailable ({type(e).__name__}); decision rules used instead"
    else:
        note = "decision rules (no LLM)"
    return T.enforce_guardrails(rules_review(case), case, check_quotes=False), note


def word_report(case: dict, rv, out_path: str) -> str:
    import datetime as dt
    return T.build_word_report({case["patient_id"]: case}, {case["patient_id"]: rv}, out_path, dt.date.today(),
                               agent_used=rv.decision != "model_evidence_only")
