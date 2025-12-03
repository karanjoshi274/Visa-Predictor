import os
import re
import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from .preprocess import get_calibrator

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODELS_DIR, "xgb_final.json")

_model = None
_calibrator = get_calibrator()


def load_model():
    """Load and cache the XGBoost model."""
    global _model
    if _model is None:
        bst = xgb.Booster()
        bst.load_model(MODEL_PATH)
        _model = bst
    return _model


def predict_proba_from_df(X):

    import re
    bst = load_model()

    def clean_value(v):
        if isinstance(v, (int, float, np.number)):
            return float(v)
        s = str(v).strip().upper()
        s = s.replace('[', '').replace(']', '').replace('"', '').replace("'", '').replace(',', '').strip()

        if s in ("Y", "YES", "TRUE", "1"):
            return 1.0
        if s in ("N", "NO", "FALSE", "0"):
            return 0.0

        s = re.sub(r"[^0-9E\.\-\+]", "", s)
        try:
            return float(s)
        except Exception:
            return 0.0

    for col in X.columns:
        X[col] = X[col].apply(clean_value)

    X = X.fillna(0.0).astype(float)

    try:
        dmat = xgb.DMatrix(X, feature_names=list(X.columns))
        pred = bst.predict(dmat)
        prob = float(pred[0]) if isinstance(pred, (list, np.ndarray)) else float(pred)
    except Exception as e:
        print("⚠️ Prediction error:", e)
        prob = 0.0

    feature_impact = {}
    try:
        explainer = shap.TreeExplainer(bst)
        shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list):  
            shap_values = shap_values[0]

        shap_values = np.array(shap_values, dtype=float)


        mean_abs = np.abs(shap_values).mean(axis=0)
        feature_impact = dict(sorted(zip(X.columns, mean_abs), key=lambda x: -x[1]))

    except Exception as e:
        print("⚠️ SHAP fallback:", e)
 
        try:
            importance_dict = bst.get_score(importance_type='gain')
            feature_impact = dict(sorted(importance_dict.items(), key=lambda x: -x[1]))
        except Exception:
            feature_impact = {col: 0.0 for col in X.columns}

    return prob, feature_impact





def generate_recommendations(feature_impact):

    if not feature_impact:
        return ["No major improvement areas detected."]

    top_features = list(feature_impact.keys())[:5]
    suggestions = []

    for feat in top_features:
        fname = feat.upper()
        if "WAGE" in fname:
            suggestions.append("Increase offered wage to at least match or exceed the prevailing wage.")
        elif "FULL_TIME" in fname:
            suggestions.append("Ensure the role is full-time to improve visa approval chances.")
        elif "H_1B_DEPENDENT" in fname:
            suggestions.append("Try applying through an employer that is not H-1B dependent.")
        elif "WILLFUL_VIOLATOR" in fname:
            suggestions.append("Ensure employer has a clean compliance record and no violations.")
        elif "AGREE_TO_LC" in fname:
            suggestions.append("Make sure to agree to all Labor Condition (LC) statements before filing.")
        elif "DURATION" in fname:
            suggestions.append("Consider requesting a longer employment duration for better stability.")
        elif "CHANGE_EMPLOYER" in fname:
            suggestions.append("Minimize frequent employer changes to show job consistency.")
        elif "NEW_EMPLOYMENT" in fname:
            suggestions.append("Provide strong documentation for new employment cases.")
        elif "CONTINUED_EMPLOYMENT" in fname:
            suggestions.append("Highlight continued employment history for stronger approval odds.")
        elif "STATE" in fname:
            suggestions.append("Double-check employer and worksite state consistency.")
        else:
            suggestions.append(f"Review and verify accuracy for '{feat.replace('_', ' ').title()}'.")

    return suggestions or ["Profile looks strong — maintain accurate documentation."]

