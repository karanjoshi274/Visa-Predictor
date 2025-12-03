import os
import pandas as pd
import numpy as np

BASE = os.path.dirname(os.path.dirname(__file__))

DATA = os.path.join(BASE, "data", "H1B_LCA_Disclosure_Data.csv")
OUT  = os.path.join(BASE, "data", "cache")

os.makedirs(OUT, exist_ok=True)

def to_yearly(value, unit):
    if pd.isna(value):
        return np.nan
    try:
        v = float(value)
    except:
        return np.nan

    u = str(unit or "").strip().lower()
    if u in ("year", "yr", "annual"):   return v
    if u in ("hour", "hr"):             return v * 2080
    if u in ("week", "wk"):             return v * 52
    if u in ("month", "mo"):            return v * 12
    if u in ("day", "d"):               return v * 260
    if u in ("bi-weekly", "biweekly"):  return v * 26
    return np.nan

print("📥 Reading:", DATA)

usecols = [
    "SOC_CODE","WORKSITE_STATE","WAGE_RATE_OF_PAY_FROM",
    "WAGE_UNIT_OF_PAY","JOB_TITLE"
]

df = pd.read_csv(DATA, usecols=usecols, low_memory=False)

df["SOC_CODE"] = df["SOC_CODE"].astype(str).str.strip()
df["WORKSITE_STATE"] = df["WORKSITE_STATE"].astype(str).str.strip()

df["WAGE_YR"] = df.apply(
    lambda r: to_yearly(r["WAGE_RATE_OF_PAY_FROM"], r["WAGE_UNIT_OF_PAY"]),
    axis=1,
)

df = df.dropna(subset=["SOC_CODE","WORKSITE_STATE","WAGE_YR"])
df = df[(df["WAGE_YR"] > 5000) & (df["WAGE_YR"] < 1_000_000)]

print("🧮 Aggregating medians…")

agg = (
    df.groupby(["SOC_CODE","WORKSITE_STATE"])
      .agg(
          median_wage=("WAGE_YR","median"),
          p25=("WAGE_YR", lambda x: np.percentile(x,25)),
          p75=("WAGE_YR", lambda x: np.percentile(x,75)),
          n=("WAGE_YR","size")
      )
      .reset_index()
)

out_path = os.path.join(OUT, "wage_index.parquet")
agg.to_parquet(out_path, index=False)

print("✅ Wrote:", out_path)
print("✅ Rows:", len(agg))
