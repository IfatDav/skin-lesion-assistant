"""Shared paths, label space and mappings for the image model."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("SKIN_DATA_ROOT", REPO_ROOT.parent / "DATA")).resolve()
META_DIR = DATA_ROOT / "metadata"
# Google Drive adds a download timestamp to the folder names (isic_512-2026...-1-001/isic_512), so search for them.
ISIC_DIR = next(iter(sorted(DATA_ROOT.glob("isic_512*/isic_512"))), DATA_ROOT / "isic_512")
PAD_DIRS = sorted(DATA_ROOT.glob("pad_original*/pad_original")) or [DATA_ROOT / "pad_original"]
PAD_RESIZED_DIR = REPO_ROOT / "data" / "processed" / "pad_512"

# Labels for the ISIC images, pulled from the ISIC API (not part of the Drive download), kept in the repo.
ISIC_META_CSV = REPO_ROOT / "data" / "metadata" / "isic_pool_api_metadata.csv"
SPLITS_DIR = REPO_ROOT / "data" / "splits"
INDEX_CSV = SPLITS_DIR / "image_index.csv"
MODELS_DIR = REPO_ROOT / "models"
REPORTS_DIR = REPO_ROOT / "reports"

SEED = 42
IMG_SIZE = 448

# Order matters: model output index i is CLASSES[i].
CLASSES = ["MEL", "SCC", "BCC", "AK", "NEV", "BKL"]
CANCER = {"MEL", "SCC", "BCC"}           # charter M2 definition
TIER = {"MEL": "urgent", "SCC": "urgent",
        "BCC": "refer", "AK": "refer",
        "NEV": "routine", "BKL": "routine"}

# ISIC archive `diagnosis_3` -> unified class. The MILK10k challenge CSV is not used:
# its AKIEC column merges actinic keratosis (pre-cancer) with SCC in situ (cancer).
ISIC_TO_CLASS = {
    "Melanoma Invasive": "MEL", "Melanoma in situ": "MEL", "Melanoma metastasis": "MEL", "Melanoma, NOS": "MEL",
    "Basal cell carcinoma": "BCC",
    "Squamous cell carcinoma, Invasive": "SCC", "Squamous cell carcinoma in situ": "SCC",
    "Squamous cell carcinoma, NOS": "SCC", "Keratoacanthoma": "SCC",
    "Solar or actinic keratosis": "AK",
    "Nevus": "NEV",
    "Seborrheic keratosis": "BKL", "Lichen planus like keratosis": "BKL", "Solar lentigo": "BKL",
}
PAD_TO_CLASS = {"MEL": "MEL", "SCC": "SCC", "BCC": "BCC", "ACK": "AK", "NEV": "NEV", "SEK": "BKL"}

MSK_ATTRIBUTION = "Memorial Sloan Kettering Cancer Center"
