import os
import time
import pandas as pd
import numpy as np
from datetime import datetime
from .preprocess import prepare_input_dict
from .model_utils import predict_proba_from_df, generate_recommendations
from .scorecard import compute_strength_score

def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().upper().replace(" ", "_") for c in df.columns]
    return df

def _safe_get(row, keys, default=None):
    for k in keys:
        if k in row and pd.notnull(row[k]) and str(row[k]).strip() != "":
            return row[k]
    return default

def _to_float(x, default=0.0):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return default
        return float(str(x).replace(',', '').strip())
    except Exception:
        return default

def _parse_date(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    try:
        return pd.to_datetime(str(s))
    except Exception:
        return None

def process_bulk_csv(df: pd.DataFrame, export_dir: str):
    df = _norm_cols(df)
    results = []
    for idx, row in df.iterrows():
        try:
            form = {
                "VISA_CLASS": _safe_get(row, ["VISA_CLASS", "VISA", "VISA_CLASS_"]),
                "JOB_TITLE": _safe_get(row, ["JOB_TITLE", "TITLE"]),
                "SOC_CODE": _safe_get(row, ["SOC_CODE"]),
                "SOC_TITLE": _safe_get(row, ["SOC_TITLE"]),
                "EMPLOYER_NAME": _safe_get(row, ["EMPLOYER_NAME", "EMPLOYER", "COMPANY"]),
                "EMPLOYER_STATE": _safe_get(row, ["EMPLOYER_STATE", "EMPLOYER_ST", "EMPLOYERSTATE"]),
                "WORKSITE_STATE": _safe_get(row, ["WORKSITE_STATE", "WORKSITE_STATE", "WORKSITE_STATE"]),
                "WORKSITE_CITY": _safe_get(row, ["WORKSITE_CITY", "WORKSITE_CITY", "WORKSITE_CITY"]),
                "FULL_TIME_POSITION": _safe_get(row, ["FULL_TIME_POSITION", "FULL_TIME", "FULLTIME_POSITION"], default="N"),
                "TOTAL_WORKER_POSITIONS": _safe_get(row, ["TOTAL_WORKER_POSITIONS", "NUM_POSITIONS", "POSITIONS"]),
                "WAGE_RATE_OF_PAY_FROM": _safe_get(row, ["WAGE_RATE_OF_PAY_FROM", "WAGE_RATE_OF_PAY", "OFFERED_WAGE", "WAGE"]),
                "WAGE_UNIT_OF_PAY": _safe_get(row, ["WAGE_UNIT_OF_PAY", "WAGE_UNIT", "WAGE_UNIT_OF_PAY"]),
                "PREVAILING_WAGE": _safe_get(row, ["PREVAILING_WAGE", "PREVAILING", "PW"]),
                "NEW_EMPLOYMENT": _safe_get(row, ["NEW_EMPLOYMENT"]),
                "CONTINUED_EMPLOYMENT": _safe_get(row, ["CONTINUED_EMPLOYMENT"]),
                "CHANGE_EMPLOYER": _safe_get(row, ["CHANGE_EMPLOYER"]),
                "H_1B_DEPENDENT": _safe_get(row, ["H_1B_DEPENDENT", "H1B_DEPENDENT"]),
                "WILLFUL_VIOLATOR": _safe_get(row, ["WILLFUL_VIOLATOR"]),
                "AGREE_TO_LC_STATEMENT": _safe_get(row, ["AGREE_TO_LC_STATEMENT", "AGREE_TO_LC"]),
                "BEGIN_DATE": _safe_get(row, ["BEGIN_DATE"]),
                "END_DATE": _safe_get(row, ["END_DATE"]),
                "EMAIL": _safe_get(row, ["EMAIL", "EMAIL_ADDRESS", "CONTACT_EMAIL"]),
            }

            bdt = _parse_date(form.get("BEGIN_DATE"))
            edt = _parse_date(form.get("END_DATE"))
            derived = {
                "BEGIN_YEAR": int(bdt.year) if bdt is not None else 0,
                "DURATION_DAYS": int((edt - bdt).days) if (bdt is not None and edt is not None) else 0
            }

            try:
                scorecard = compute_strength_score(form, derived)
            except TypeError:
                scorecard = compute_strength_score(form, derived, offered_wage=form.get("WAGE_RATE_OF_PAY_FROM"))

            try:
                X = prepare_input_dict(form)
                prob, feature_impact = predict_proba_from_df(X)
            except Exception as e:
                prob = 0.0
                feature_impact = {}

            try:
                recs = generate_recommendations(feature_impact)
            except Exception:
                recs = []

            if isinstance(prob, (float, int)):
                pct = float(prob) * 100.0
                if pct > 75:
                    rec_label = "✅ High chance of approval"
                elif pct > 45:
                    rec_label = "⚠️ Moderate likelihood of approval"
                else:
                    rec_label = "❌ High chance of denial"
            else:
                pct = "N/A"
                rec_label = "Error in prediction"

            results.append({
                "EMPLOYER_NAME": form.get("EMPLOYER_NAME"),
                "JOB_TITLE": form.get("JOB_TITLE"),
                "OFFERED_WAGE": form.get("WAGE_RATE_OF_PAY_FROM"),
                "FULL_TIME_POSITION": form.get("FULL_TIME_POSITION"),
                "probability_%": round(pct, 2) if isinstance(pct, float) else pct,
                "recommendation": rec_label,
                "score_wage": scorecard.get("wage_score", 0),
                "score_compliance": scorecard.get("compliance_score", 0),
                "score_stability": scorecard.get("stability_score", 0),
                "score_docs": scorecard.get("documentation_score", 0),
                "score_total": scorecard.get("total_score", 0),
            })

        except Exception as exc:
            results.append({
                "EMPLOYER_NAME": _safe_get(row, ["EMPLOYER_NAME"]) or None,
                "JOB_TITLE": _safe_get(row, ["JOB_TITLE"]) or None,
                "OFFERED_WAGE": _safe_get(row, ["WAGE_RATE_OF_PAY_FROM", "OFFERED_WAGE"]) or None,
                "FULL_TIME_POSITION": _safe_get(row, ["FULL_TIME_POSITION"]) or None,
                "probability_%": "N/A",
                "recommendation": f"Error: {str(exc)}",
                "score_wage": 0,
                "score_compliance": 0,
                "score_stability": 0,
                "score_docs": 0,
                "score_total": 0,
            })

    results_df = pd.DataFrame(results)

    os.makedirs(export_dir, exist_ok=True)
    filename = f"bulk_results_{int(time.time())}.csv"
    outpath = os.path.join(export_dir, filename)
    results_df.to_csv(outpath, index=False)

    return results_df, filename
