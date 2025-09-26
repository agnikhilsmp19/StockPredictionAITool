from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PRED = ROOT / "data" / "predictions"

# Ensure folders exist
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PRED.mkdir(parents=True, exist_ok=True)
