
"""Exploratory M1 stability analysis.

Runs multiple random seeds without modifying
the frozen M1 evaluation or its original report.
"""

from pathlib import Path
import json

import pandas as pd

from src.simulate_queue import (
    simulate,
    summarize,
    SCORES,
)

OUTPUT = Path("reports/m1_stability_exploratory.json")

SEEDS = [42, 123, 456, 789, 2026]
