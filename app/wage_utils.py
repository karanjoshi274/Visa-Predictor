
import os, pandas as pd, numpy as np

CACHE = os.path.join(os.path.dirname(__file__), "..", "data", "cache", "wage_index.parquet")

def to_yearly(value, unit):
    try:
        v = float(value or 0)
    except Exception:
        return None
    u = str(unit or "").strip().lower()
    if u in ("year","yr","y","per year","annum","annual"): return v
    if u in ("hour","hr","h"):                              return v*2080
    if u in ("week","wk","w"):                              return v*52
    if u in ("month","mo","m"):                             return v*12
    if u in ("bi-weekly","biweekly"):                       return v*26
    if u in ("day","d"):                                    return v*260
    return None

_wage_df = None
def load_index():
    global _wage_df
    if _wage_df is None:
        _wage_df = pd.read_parquet(CACHE)
    return _wage_df

def compare_wage(soc_code, state, offered_value, offered_unit):
    df = load_index()
    soc = str(soc_code or "").strip()
    st  = str(state or "").strip()
    row = df[(df["SOC_CODE"]==soc) & (df["WORKSITE_STATE"]==st)]
    if row.empty:
        return {"found": False, "message": "No benchmark found for SOC/state."}

    offered_yearly = to_yearly(offered_value, offered_unit)
    if offered_yearly is None:
        return {"found": False, "message": "Invalid offered wage."}

    r = row.iloc[0].to_dict()
    pct = offered_yearly / r["median_wage"] if r["median_wage"] else None
    verdict = ("Below median" if pct and pct < 1.0 else
               "Meets median" if pct and 0.99 <= pct <= 1.05 else
               "Above median")
    return {
        "found": True,
        "offered_yearly": round(offered_yearly,2),
        "median": round(r["median_wage"],2),
        "p25": round(r["p25"],2),
        "p75": round(r["p75"],2),
        "n": int(r["n"]),
        "ratio": round(pct,3) if pct else None,
        "verdict": verdict
    }
