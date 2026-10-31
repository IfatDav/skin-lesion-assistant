"""Regression tests for shared queue-relative risk bands and the legacy generator."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.fusion import rank_band
from agent.make_daily_list import make_daily_list
from src.build_daily_cases_from_val import rank_band as bridge_rank_band

class RiskBandTests(unittest.TestCase):
    def test_boundaries(self):
        cases = [(0.0, "low"), (0.3999, "low"), (0.40, "moderate"),
                 (0.70, "high"), (0.90, "very high"), (1.0, "very high")]
        for score, expected in cases:
            self.assertEqual(rank_band(score), expected)

    def test_invalid_scores(self):
        for value in (-0.01, 1.01, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                rank_band(value)

    def test_shared_policy_import(self):
        self.assertIs(bridge_rank_band, rank_band)

    def test_legacy_daily_selection_and_unchecked_quality(self):
        patients = []
        for i in range(100):
            row = {
                "patient_id": f"P{i:03d}",
                "diagnostic": "BCC" if i < 10 else "NEV",
                "image_score": float(i), "symptom_score": float(i),
                "final_score": 0.0,  # intentionally stale and ignored
                "age": 45, "region": "FACE", "img_id": f"p{i}.png",
            }
            row.update({symptom: False for symptom in
                        ("itch", "grew", "hurt", "changed", "bleed", "elevation")})
            patients.append(row)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "cases.csv")
            output = make_daily_list(pd.DataFrame(patients), out_path=path)
            self.assertEqual(len(pd.read_csv(path)), 10)
        self.assertEqual(output.patient_id.tolist(), [f"P{i:03d}" for i in range(99, 89, -1)])
        self.assertTrue(output.risk_band.eq("very high").all())
        self.assertTrue(output.image_risk_band.eq("very high").all())
        self.assertTrue(output.symptom_risk_band.eq("very high").all())
        self.assertTrue(output.image_quality.eq("not_checked").all())
        self.assertNotIn("diagnostic", output)
        self.assertEqual(output.risk_score.iloc[0], 1.0)


if __name__ == "__main__":
    unittest.main()
