import os
import io
import pandas as pd
from typing import Dict, Any
from fastapi import FastAPI, Request, Form, UploadFile, File, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

from .preprocess import prepare_input_dict
from .model_utils import predict_proba_from_df, generate_recommendations
from .reinforcement import log_submission
from .email_utils import send_result_email
from .online_validate import validate_job_employer
from .scorecard import compute_strength_score
from .wage_utils import compare_wage
from .bulk_utils import process_bulk_csv
from .guides import suggest_from_flags
from .chatbot import chat_respond

BASE_DIR = os.path.dirname(__file__)
app = FastAPI(title="Visa Approval Predictor")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")


os.makedirs(os.path.join(BASE_DIR, "static", "exports"), exist_ok=True)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    visa_class: str = Form(None),
    job_title: str = Form(None),
    soc_code: str = Form(None),
    soc_title: str = Form(None),
    employer_name: str = Form(None),
    employer_state: str = Form(None),
    worksite_state: str = Form(None),
    worksite_city: str = Form(None),
    full_time_position: str = Form(None),
    total_worker_positions: str = Form(None),
    wage: str = Form(None),
    wage_unit: str = Form("Year"),
    prevailing_wage: str = Form(None),
    new_employment: str = Form(None),
    continued_employment: str = Form(None),
    change_employer: str = Form(None),
    h1b_dependent: str = Form(None),
    willful_violator: str = Form(None),
    agree_lc: str = Form(None),
    begin_date: str = Form(None),
    end_date: str = Form(None),
    email: str = Form(None),
):
    try:
        form = {
            "VISA_CLASS": (visa_class or "").strip(),
            "JOB_TITLE": (job_title or "").strip(),
            "SOC_CODE": (soc_code or "").strip(),
            "SOC_TITLE": (soc_title or "").strip(),
            "EMPLOYER_NAME": (employer_name or "").strip(),
            "EMPLOYER_STATE": (employer_state or "").strip(),
            "WORKSITE_STATE": (worksite_state or "").strip(),
            "WORKSITE_CITY": (worksite_city or "").strip(),
            "FULL_TIME_POSITION": (full_time_position or "").strip(),
            "TOTAL_WORKER_POSITIONS": (total_worker_positions or "").strip(),
            "WAGE_RATE_OF_PAY_FROM": (wage or "").strip(),
            "WAGE_UNIT_OF_PAY": (wage_unit or "Year").strip(),
            "PREVAILING_WAGE": (prevailing_wage or "").strip(),
            "NEW_EMPLOYMENT": (new_employment or "").strip(),
            "CONTINUED_EMPLOYMENT": (continued_employment or "").strip(),
            "CHANGE_EMPLOYER": (change_employer or "").strip(),
            "H_1B_DEPENDENT": (h1b_dependent or "").strip(),
            "WILLFUL_VIOLATOR": (willful_violator or "").strip(),
            "AGREE_TO_LC_STATEMENT": (agree_lc or "").strip(),
            "BEGIN_DATE": (begin_date or "").strip(),
            "END_DATE": (end_date or "").strip(),
        }

        X = prepare_input_dict(form)
        base_prob, feature_impact = predict_proba_from_df(X)
        begin_dt = pd.to_datetime(begin_date or "", errors="coerce")
        end_dt = pd.to_datetime(end_date or "", errors="coerce")
        derived = {
            "BEGIN_YEAR": int(begin_dt.year) if pd.notnull(begin_dt) else 0,
            "DURATION_DAYS": int((end_dt - begin_dt).days) if (pd.notnull(begin_dt) and pd.notnull(end_dt)) else 0,
        }
        scorecard = compute_strength_score(form, derived)
        notes = []
        rule_suggestions = []
        adjusted_prob = base_prob

        def yn(v): return str(v).strip().lower() in ("y", "yes", "true", "1")
        try:
            wage_val = float(wage or 0)
            prev_wage_val = float(prevailing_wage or 0)
            if prev_wage_val > 0 and wage_val < prev_wage_val:
                adjusted_prob *= 0.75
                notes.append("Offered wage is below prevailing wage (-25%).")
                rule_suggestions.append("Increase offered wage closer to or above the prevailing wage.")
            elif prev_wage_val > 0 and wage_val >= 1.2 * prev_wage_val:
                adjusted_prob *= 1.05
                notes.append("Offered wage significantly exceeds prevailing wage (+5%).")
        except Exception:
            pass

        if not yn(full_time_position):
            adjusted_prob *= 0.8
            notes.append("Not a full-time position (-20%).")
            rule_suggestions.append("Convert to a full-time role if possible.")

        if yn(h1b_dependent):
            adjusted_prob *= 0.85
            notes.append("Employer is H-1B dependent (-15%).")
            rule_suggestions.append("Reduce dependency on H-1B workforce or justify dependency clearly.")

        if yn(willful_violator):
            adjusted_prob *= 0.8
            notes.append("Employer flagged as willful violator (-20%).")
            rule_suggestions.append("Ensure full compliance and file corrective documentation.")

        if not yn(agree_lc):
            adjusted_prob *= 0.7
            notes.append("Labor Condition Statement not agreed (-30%).")
            rule_suggestions.append("Agree to LC statement before filing.")

        adjusted_prob = max(0.0, min(adjusted_prob, 1.0))

        flags: Dict[str, Any] = {}
        try:
            flags["wage_below_prev"] = (float(prevailing_wage or 0) > 0) and (float(wage or 0) < float(prevailing_wage or 0))
        except Exception:
            flags["wage_below_prev"] = False
        flags["not_full_time"] = not yn(full_time_position)
        flags["h1b_dependent"] = yn(h1b_dependent)
        flags["willful_violator"] = yn(willful_violator)
        flags["no_lc_agree"] = not yn(agree_lc)
        try:
            months = int(max(0, (end_dt - begin_dt).days // 30)) if (pd.notnull(begin_dt) and pd.notnull(end_dt)) else 0
        except Exception:
            months = 0
        flags["short_duration"] = bool(months and months < 12)

        guide_snippets = suggest_from_flags(flags)

        shap_suggestions = generate_recommendations(feature_impact)
        all_suggestions = (rule_suggestions or []) + (shap_suggestions or []) + (guide_snippets or [])

        if adjusted_prob > 0.75:
            recommendation = "✅ High chance of approval"
        elif adjusted_prob > 0.45:
            recommendation = "⚠️ Moderate likelihood of approval"
        else:
            recommendation = "❌ High chance of denial"

        validation_notes = validate_job_employer(job_title or "", employer_state or "OK")
        log_submission(form, adjusted_prob)

        subj = "Visa Predictor: Your Result"
        body = (
            f"Estimated approval probability: {adjusted_prob*100:.2f}%\n"
            f"Recommendation: {recommendation}\n\n"
            f"Key Factors:\n- " + "\n- ".join(notes or ["Basic validations OK."]) + "\n\n"
            f"Scorecard:\n"
            f"- Wage Competitiveness: {scorecard['wage_score']:.1f}/100\n"
            f"- Compliance Posture: {scorecard['compliance_score']:.1f}/100\n"
            f"- Employment Stability: {scorecard['stability_score']:.1f}/100\n"
            f"- Documentation Completeness: {scorecard['documentation_score']:.1f}/100\n"
            f"- Total Strength: {scorecard['total_score']:.1f}/100\n\n"
            f"Suggestions:\n- " + "\n- ".join(all_suggestions or ["Everything looks good!"])
        )
        email_sent = send_result_email(email or "", subj, body)

        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "probability": f"{adjusted_prob*100:.2f}",
                "recommendation": recommendation,
                "notes": validation_notes or "Basic validations OK.",
                "details": notes or ["Basic validations OK."],
                "feature_impact": feature_impact or {},
                "suggestions": all_suggestions or ["Everything looks good!"],
                "scorecard": scorecard,
                "email": email,
                "email_sent": email_sent,
            },
        )

    except Exception as e:
        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "probability": "N/A",
                "recommendation": "Prediction failed.",
                "notes": str(e),
                "details": [],
                "feature_impact": {},
                "suggestions": [],
                "scorecard": None,
                "email": email,
                "email_sent": "No",
            },
        )

@app.get("/wage", response_class=HTMLResponse)
async def wage_form(request: Request):
    return templates.TemplateResponse("wage.html", {"request": request, "result": None})

@app.post("/wage", response_class=HTMLResponse)
async def wage_post(
    request: Request,
    soc_code: str = Form(""),
    worksite_state: str = Form(""),
    offered_wage: str = Form(""),
    wage_unit: str = Form("Year"),
):
    result = compare_wage((soc_code or "").strip(), (worksite_state or "").strip(), (offered_wage or "").strip(), (wage_unit or "Year").strip())
    return templates.TemplateResponse("wage.html", {
        "request": request,
        "result": result,
        "soc_code": soc_code,
        "worksite_state": worksite_state,
        "offered_wage": offered_wage,
        "wage_unit": wage_unit
    })


@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/chat/message")
async def chat_message(payload: dict = Body(...)):
    user_text = str(payload.get("message", "")).strip()
    reply, actions = chat_respond(user_text)
    return {"reply": reply, "actions": actions}

@app.get("/bulk", response_class=HTMLResponse)
async def bulk_form(request: Request):
    return templates.TemplateResponse("bulk.html", {"request": request, "preview": None})

@app.post("/bulk", response_class=HTMLResponse)
async def bulk_post(request: Request, file: UploadFile = File(...)):
    try:
        content = await file.read()
        try:
            df = pd.read_csv(io.BytesIO(content))
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(content), encoding="latin-1")

        export_dir = os.path.join(BASE_DIR, "static", "exports")
        results_df, filename = process_bulk_csv(df, export_dir)
        preview = results_df.head(20).to_dict(orient="records")
        download_url = f"/static/exports/{filename}"
        return templates.TemplateResponse("bulk.html", {
            "request": request,
            "preview": preview,
            "columns": list(results_df.columns),
            "download": download_url
        })
    except Exception as e:
        return templates.TemplateResponse("bulk.html", {"request": request, "error": str(e), "preview": None})


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
