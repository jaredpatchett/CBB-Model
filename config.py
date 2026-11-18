"""
Central config for the CBB model: API keys from the environment (a .env file
locally, repo secrets in GitHub Actions), paths and the season calendar.
Mirrors the CFB model's config.py.
"""
import os
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()


def _clean(v):
    return v.strip() if v else v


CBBD_API_KEY = _clean(os.environ.get("CBBD_API_KEY"))
ODDS_API_KEY = _clean(os.environ.get("ODDS_API_KEY"))

CBBD_BASE_URL = "https://api.collegebasketballdata.com"
ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"
ODDS_API_SPORT = "basketball_ncaab"

DATA_RAW_DIR = "data/raw"
DATA_PROCESSED_DIR = "data/processed"
DATA_CURRENT_DIR = "data/current"
MODELS_DIR = "models"
DOCS_DATA_DIR = "docs/data"

# Set CBB_DEMO=1 to run the whole pipeline on synthetic data
# (scripts/demo_data.py) with no API keys -- for building and testing only.
DEMO = os.environ.get("CBB_DEMO") == "1"


def require_keys(*names):
    if DEMO:
        return
    missing = [n for n in names if not globals().get(n)]
    if missing:
        raise EnvironmentError(f"Missing API key(s): {', '.join(missing)}. Set them in .env or as GitHub secrets.")


def current_season(now: datetime = None) -> int:
    """CBBD labels a season by the year it ENDS (2026-27 season = 2027).
    November-April belongs to the season ending next spring."""
    now = now or datetime.now(timezone.utc)
    return now.year + 1 if now.month >= 8 else now.year
