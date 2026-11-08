"""Code makes the decisions; a verified finding from the agent can only raise a `proceed`. No LLM needed."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

import m4_eval as m4
import triage_agent as t


def case(**kw):
    base = {"patient_id": "P-1", "rank": 1, "risk_score": 0.9, "risk_band": "high", "image_risk_band": "high",
            "symptom_risk_band": "high", "age": 60, "body_site": "back",
            "symptoms": {s: "no" for s in t.SYMPTOMS}, "patient_note": "", "image_path": "p.jpg",
            "image_quality": "usable", "image_issues": []}
    return {**base, **kw}


def contradiction(quote, symptom="bleed", note_says="yes"):
    return t.NoteFinding(kind="contradiction", quote=quote, symptom=symptom, note_says=note_says)


def run(c, **kw):
    return t.enforce_guardrails(t.AgentReport(patient_id=c["patient_id"], **kw), c, check_quotes=False)


class CodeDecides(unittest.TestCase):
    def test_same_as_the_if_else_baseline(self):
        for s in m4.SCENARIOS:
            d, b = t.decide(s["case"]), m4.if_else(s["case"])
            for key in ("decision", "follow_up_question", "retake_instruction"):
                self.assertEqual(d[key], b[key], (s["id"], key))

    def test_fifteen_scenarios_without_the_agent(self):
        rows = m4.run_system()
        self.assertEqual([r["id"] for r in rows if not r["pass"]], [9, 11, 12])

    def test_agent_schema_has_no_decision_score_or_rank(self):
        self.assertFalse({"decision", "risk_score", "rank"} & set(t.AgentReport.model_fields))

    def test_decision_in_agent_json_is_ignored(self):
        class Raw:
            raw = '{"patient_id": "P-1", "decision": "manual_review", "rank": 9}'
        c = case()
        self.assertEqual(t.enforce_guardrails(t.parse_review(Raw()), c, check_quotes=False).decision, "proceed")


class Findings(unittest.TestCase):
    NOTE = "last week there was something on my shirt"

    def test_verified_contradiction_raises_proceed(self):
        c = case(patient_note=self.NOTE)
        r = run(c, note_findings=[contradiction("something on my shirt")])
        self.assertEqual(r.decision, "ask_patient")
        self.assertIn("האם יש דימום", r.follow_up_question)
        self.assertEqual(t.OUTBOX["P-1"]["type"], "question")

    def test_new_information_flags_proceed(self):
        c = case(patient_note=self.NOTE)
        r = run(c, note_findings=[t.NoteFinding(kind="new_information", quote="on my shirt")])
        self.assertEqual(r.decision, "proceed_with_flag")

    def test_quote_not_in_note_is_ignored(self):
        c = case(patient_note=self.NOTE)
        r = run(c, note_findings=[contradiction("it bled a lot")])
        self.assertEqual(r.decision, "proceed")
        self.assertTrue(any("could not be verified" in f for f in r.flags))

    def test_form_agrees_with_the_note_is_ignored(self):
        c = case(patient_note=self.NOTE, symptoms={**case()["symptoms"], "bleed": "yes"})
        self.assertEqual(run(c, note_findings=[contradiction("on my shirt")]).decision, "proceed")

    def test_unknown_form_answer_is_not_a_contradiction(self):
        c = case(patient_note=self.NOTE, symptoms={**case()["symptoms"], "bleed": "unknown"})
        self.assertEqual(run(c, note_findings=[contradiction("on my shirt")]).decision, "proceed")

    def test_findings_never_change_other_decisions(self):
        for kw, expected in [({"age": 16}, "manual_review"),
                             ({"image_quality": "unusable"}, "request_retake"),
                             ({"image_quality": "borderline", "risk_band": "very high"}, "proceed_with_flag")]:
            c = case(patient_note=self.NOTE, **kw)
            r = run(c, note_findings=[contradiction("on my shirt")])
            self.assertEqual(r.decision, expected)
            self.assertTrue(any("on my shirt" in f for f in r.flags))   # the dermatologist still sees it


class Texts(unittest.TestCase):
    def test_agent_text_is_used_when_it_passes(self):
        c = case(image_quality="unusable", image_issues=["too dark"])
        self.assertEqual(run(c, retake_instruction="צלמו ליד חלון.").retake_instruction, "צלמו ליד חלון.")

    def test_reassuring_question_is_replaced_decision_unchanged(self):
        c = case(symptom_risk_band="low", image_risk_band="very high")
        r = run(c, follow_up_question="הנגע בטוח?")
        self.assertEqual(r.decision, "ask_patient")
        self.assertEqual(r.follow_up_question, "האם חל שינוי בנגע לאחרונה?")

    def test_reassuring_summary_is_replaced_decision_unchanged(self):
        r = run(case(), evidence_summary="This looks benign.")
        self.assertEqual(r.decision, "proceed")
        self.assertNotIn("benign", r.evidence_summary)

    def test_reason_comes_from_code(self):
        r = run(case(), evidence_summary="Anything.")
        self.assertEqual(r.decision_reason, t.decide(case())["reason"])

    def test_no_message_without_ask_or_retake(self):
        c = case()
        run(c, follow_up_question="האם יש דימום?", retake_instruction="צלמו שוב.")
        self.assertNotIn("P-1", t.OUTBOX)


if __name__ == "__main__":
    unittest.main()
