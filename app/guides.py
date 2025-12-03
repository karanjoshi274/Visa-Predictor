import os
import json
from typing import Dict, List

BASE_DIR = os.path.dirname(__file__)
GUIDE_PATH = os.path.join(BASE_DIR, "..", "data", "guides.json")

_GUIDES = None

def _load() -> Dict[str, List[str]]:
    global _GUIDES
    if _GUIDES is None:
        try:
            with open(GUIDE_PATH, "r", encoding="utf-8") as f:
                _GUIDES = json.load(f)
        except Exception:
            _GUIDES = {}
    return _GUIDES

def suggest_from_flags(flags: Dict[str, bool]) -> List[str]:

    g = _load()
    picks = []
    if flags.get("wage_below_prev"):
        picks += g.get("wage_below_prev", [])
    if flags.get("not_full_time"):
        picks += g.get("not_full_time", [])
    if flags.get("h1b_dependent"):
        picks += g.get("h1b_dependent", [])
    if flags.get("willful_violator"):
        picks += g.get("willful_violator", [])
    if flags.get("no_lc_agree"):
        picks += g.get("no_lc_agree", [])
    if flags.get("short_duration"):
        picks += g.get("short_duration", [])
    # de-duplicate while preserving order
    seen, dedup = set(), []
    for s in picks:
        if s not in seen:
            dedup.append(s)
            seen.add(s)
    return dedup[:6]  # keep it concise
