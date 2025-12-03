import os, joblib, json, pandas as pd, numpy as np
from dateutil import parser


BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

def load_artifacts():
    enc_path = os.path.join(MODELS_DIR, "feature_encoder.joblib")
    scaler_path = os.path.join(MODELS_DIR, "feature_scaler.joblib")
    meta_path = os.path.join(MODELS_DIR, "metadata.json")
    calib_path = os.path.join(MODELS_DIR, "prob_calibrator.joblib")

    enc = joblib.load(enc_path) if os.path.exists(enc_path) else {}
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    calibrator = joblib.load(calib_path) if os.path.exists(calib_path) else None
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return enc, scaler, meta, calibrator

ENCODERS, SCALER, METADATA, CALIBRATOR = load_artifacts()
FEATURE_COLUMNS = METADATA.get("features", [])

def normalize_yesno(v):
    if v is None:
        return 0
    v = str(v).strip().lower()
    return 1 if v in ("y", "yes", "true", "1") else 0

def safe_float(v):

    try:
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).strip().replace('[', '').replace(']', '')
        return float(s)
    except Exception:
        return 0.0


def prepare_input_dict(form: dict):

    row = {}

    for col in FEATURE_COLUMNS:
        row[col] = form.get(col, "MISSING")


    for n in ["TOTAL_WORKER_POSITIONS", "WAGE_RATE_OF_PAY_FROM", "PREVAILING_WAGE"]:
        try:
            row[n] = safe_float(form.get(n, 0))
        except Exception:
            row[n] = 0.0

    for yn in [
        "FULL_TIME_POSITION", "NEW_EMPLOYMENT", "CONTINUED_EMPLOYMENT",
        "CHANGE_EMPLOYER", "H_1B_DEPENDENT", "WILLFUL_VIOLATOR",
        "AGREE_TO_LC_STATEMENT"
    ]:
        row[yn] = normalize_yesno(form.get(yn))

    try:
        b = form.get("BEGIN_DATE", "")
        e = form.get("END_DATE", "")
        bd = parser.parse(b) if b else None
        ed = parser.parse(e) if e else None
        if bd:
            row["BEGIN_YEAR"] = bd.year
            row["BEGIN_MONTH"] = bd.month
        else:
            row["BEGIN_YEAR"] = row["BEGIN_MONTH"] = 0
        if ed:
            row["END_YEAR"] = ed.year
            row["END_MONTH"] = ed.month
        else:
            row["END_YEAR"] = row["END_MONTH"] = 0
        if bd and ed:
            row["DURATION_DAYS"] = (ed - bd).days
        else:
            row["DURATION_DAYS"] = 0
    except Exception:
        row["BEGIN_YEAR"] = row["BEGIN_MONTH"] = row["END_YEAR"] = row["END_MONTH"] = row["DURATION_DAYS"] = 0

    X = pd.DataFrame([row])

    for col, mapping in ENCODERS.items():
        if col in X.columns:
            val = str(X.at[0, col])
            X[col] = mapping.get(val, mapping.get("MISSING", 0))

    numeric_cols = [
        "TOTAL_WORKER_POSITIONS", "WAGE_RATE_OF_PAY_FROM", "PREVAILING_WAGE",
        "DURATION_DAYS", "BEGIN_YEAR", "BEGIN_MONTH", "END_YEAR", "END_MONTH"
    ]
    if SCALER and all(c in X.columns for c in numeric_cols):
        X[numeric_cols] = SCALER.transform(X[numeric_cols])


    for f in FEATURE_COLUMNS:
        if f not in X.columns:
            X[f] = 0

    return X[FEATURE_COLUMNS]

def get_calibrator():
    return CALIBRATOR
