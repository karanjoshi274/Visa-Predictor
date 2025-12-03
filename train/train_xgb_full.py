import os, json, pandas as pd, numpy as np
from collections import Counter
import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV

CSV_PATH = os.path.join("data", "H1B_LCA_Disclosure_Data.csv")
OUT_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)

FEATURE_COLUMNS = [
    "VISA_CLASS", "JOB_TITLE", "SOC_CODE", "SOC_TITLE",
    "EMPLOYER_NAME", "EMPLOYER_STATE", "WORKSITE_STATE", "WORKSITE_CITY",
    "FULL_TIME_POSITION", "TOTAL_WORKER_POSITIONS",
    "WAGE_RATE_OF_PAY_FROM", "WAGE_UNIT_OF_PAY", "PREVAILING_WAGE",
    "NEW_EMPLOYMENT", "CONTINUED_EMPLOYMENT", "CHANGE_EMPLOYER",
    "H_1B_DEPENDENT", "WILLFUL_VIOLATOR", "AGREE_TO_LC_STATEMENT",
    "BEGIN_DATE", "END_DATE"
]
TARGET_COL = "CASE_STATUS"

def preprocess(df):
    df = df.copy()
    df = df.dropna(subset=[TARGET_COL])
    df[TARGET_COL] = df[TARGET_COL].astype(str).str.lower().eq("certified").astype(int)
    # Duration
    df["BEGIN_DATE"] = pd.to_datetime(df["BEGIN_DATE"], errors="coerce")
    df["END_DATE"] = pd.to_datetime(df["END_DATE"], errors="coerce")
    df["DURATION_DAYS"] = (df["END_DATE"] - df["BEGIN_DATE"]).dt.days.fillna(0)
    df["BEGIN_YEAR"] = df["BEGIN_DATE"].dt.year.fillna(0)
    df["BEGIN_MONTH"] = df["BEGIN_DATE"].dt.month.fillna(0)
    df["END_YEAR"] = df["END_DATE"].dt.year.fillna(0)
    df["END_MONTH"] = df["END_DATE"].dt.month.fillna(0)

    keep_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    df = df[keep_cols + ["DURATION_DAYS", "BEGIN_YEAR", "BEGIN_MONTH", "END_YEAR", "END_MONTH", TARGET_COL]]
    df = df.fillna("MISSING")
    return df

def encode_and_scale(df):
    print("Encoding and scaling features...")
    encoders = {}
    X = pd.DataFrame()
    cat_cols = [
        "VISA_CLASS", "JOB_TITLE", "SOC_CODE", "SOC_TITLE",
        "EMPLOYER_NAME", "EMPLOYER_STATE", "WORKSITE_STATE", "WORKSITE_CITY",
        "FULL_TIME_POSITION", "WAGE_UNIT_OF_PAY",
        "NEW_EMPLOYMENT", "CONTINUED_EMPLOYMENT", "CHANGE_EMPLOYER",
        "H_1B_DEPENDENT", "WILLFUL_VIOLATOR", "AGREE_TO_LC_STATEMENT"
    ]

    num_cols = [
        "TOTAL_WORKER_POSITIONS", "WAGE_RATE_OF_PAY_FROM", "PREVAILING_WAGE",
        "DURATION_DAYS", "BEGIN_YEAR", "BEGIN_MONTH", "END_YEAR", "END_MONTH"
    ]

    for col in cat_cols:
        if col not in df.columns:
            continue
        vals = df[col].astype(str).fillna("MISSING")
        uniq = vals.value_counts().index.tolist()
        mapping = {v: i for i, v in enumerate(uniq)}
        encoders[col] = mapping
        X[col] = vals.map(mapping).astype(int)

    for col in num_cols:
        if col in df.columns:
            X[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    for date_col in ["BEGIN_DATE", "END_DATE"]:
        if date_col in X.columns:
            X = X.drop(columns=[date_col])

    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    print(f"Encoded {len(cat_cols)} categorical + {len(num_cols)} numeric features.")
    return X, encoders, scaler


def main():
    print("Loading CSV...")
    df = pd.read_csv(CSV_PATH, low_memory=False)
    df = preprocess(df)
    print("Preprocessing done.")
    X, enc, scaler = encode_and_scale(df)
    y = df[TARGET_COL].astype(int)
    print("Encoding done.")


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    print("Training XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        tree_method="hist",
        eval_metric="auc",
        use_label_encoder=False,
        n_jobs=-1
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)

    print("Calibrating probabilities...")
    calibrator = CalibratedClassifierCV(estimator=model, method="sigmoid", cv="prefit")
    calibrator.fit(X_test, y_test)

    print("Saving artifacts...")
    model.save_model(os.path.join(OUT_DIR, "xgb_final.json"))
    joblib.dump(enc, os.path.join(OUT_DIR, "feature_encoder.joblib"))
    joblib.dump(scaler, os.path.join(OUT_DIR, "feature_scaler.joblib"))
    joblib.dump(calibrator, os.path.join(OUT_DIR, "prob_calibrator.joblib"))
    metadata = {"features": list(X.columns)}
    with open(os.path.join(OUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("✅ Training complete. Model ready.")

if __name__ == "__main__":
    main()
