# app/chatbot.py
import re

HELP_LINKS = {
    "single": "/",
    "wage": "/wage",
    "bulk": "/bulk",
    "chat": "/chat"
}

FAQ = [
    (r"\bhello|hi|hey\b", "Hi! I can explain your prediction, suggest improvements, compare wages, or run bulk checks. What would you like to do?"),
    (r"\bhow (does|do) (it|this|model) work\b", 
     "We trained an XGBoost model on public LCA disclosures. We encode job, employer/worksite, wages, dates, and employment flags; we calibrate probabilities and visualize top drivers. You can also see a scorecard and wage comparator."),
    (r"\bwhat (affects|impacts) (approval|result)\b",
     "Common drivers include offered wage vs prevailing wage, full-time status, employer compliance (H-1B dependent / willful violator), employment continuity, and duration. Use the wage comparator and our suggestions to improve."),
    (r"\bprivacy|data (use|usage|policy)\b",
     "We only use the details you submit to compute the prediction and email you results if requested. Identifiers can be anonymized in logs for model improvement. Avoid sharing sensitive personal info."),
    (r"\bsupported visas?\b|\bonly h-?1b\b",
     "Right now we focus on H-1B LCA outcomes. The interface supports other visa classes in the form, but predictions are optimized for H-1B."),
    (r"\bemail (not|didn'?t) send|mail issue\b",
     "Please verify the email address format and email configuration (.env SMTP settings). If issues persist, we’ll still show the full result on screen."),
]

SUGGESTIONS_GENERAL = [
    "Compare your offered wage with market medians here → /wage",
    "Run multiple scenarios via CSV upload here → /bulk",
    "Ask: “How can I improve my chances?” for personalized tips."
]

IMPROVE_TIPS = [
    "Raise offered wage to match or exceed the prevailing wage.",
    "Convert to a full-time role if possible.",
    "If employer is H-1B dependent, strengthen justifications and compliance documentation.",
    "Agree to LC statements and ensure documentation completeness.",
    "Aim for at least a 12-month duration for stability where feasible."
]

def _detect_intent(user_text: str):
    s = user_text.lower().strip()

    # Direct intents
    if "wage" in s and ("compare" in s or "median" in s or "prevailing" in s):
        return "wage_help"
    if "bulk" in s or "csv" in s or "upload" in s:
        return "bulk_help"
    if "improve" in s or "increase" in s or "boost" in s or "chance" in s:
        return "improve"
    if "explain" in s or "why" in s and ("low" in s or "result" in s or "score" in s or "prob" in s):
        return "explain"
    if "help" in s or "what can you do" in s:
        return "capability"

    # FAQ patterns
    for pat, _ in FAQ:
        if re.search(pat, s):
            return "faq"

    return "fallback"


def _run_faq(user_text: str):
    for pat, answer in FAQ:
        if re.search(pat, user_text, flags=re.IGNORECASE):
            return answer
    # default:
    return ("I can answer questions about how the model works, what affects approval, "
            "privacy, and current visa support. Try: “What affects approval?”")


def _extract_slots_for_wage(user_text: str):
    """
    Heuristic slot extraction to help users jump into /wage with prefilled clues.
    Example matches:
      - SOC code like 15-1252 or 15.1252
      - State as two letters (FL, CA, TX)
      - Wage as a number
    """
    soc = None
    m_soc = re.search(r"\b(\d{2})[-\.](\d{4})\b", user_text)
    if m_soc:
        soc = f"{m_soc.group(1)}-{m_soc.group(2)}"

    state = None
    m_st = re.search(r"\b([A-Z]{2})\b", user_text.upper())
    if m_st:
        state = m_st.group(1)

    wage = None
    m_w = re.search(r"\b(\d{2,6})(?:\.\d+)?\b", user_text.replace(",", ""))
    if m_w:
        wage = m_w.group(1)

    return soc, state, wage


def chat_respond(user_text: str):
    intent = _detect_intent(user_text)

    if intent == "wage_help":
        soc, state, wage = _extract_slots_for_wage(user_text)
        hint = []
        if soc:   hint.append(f"SOC={soc}")
        if state: hint.append(f"STATE={state}")
        if wage:  hint.append(f"WAGE={wage}")
        hint_str = (" (detected: " + ", ".join(hint) + ")") if hint else ""
        reply = (
            "You can compare your offered wage with market medians and prevailing baselines. "
            f"Open the wage comparator here → {HELP_LINKS['wage']}{hint_str}.\n"
            "Provide SOC code (e.g., 15-1252), worksite state, and offered wage & unit."
        )
        actions = [{"label": "Open Wage Comparator", "href": HELP_LINKS["wage"]}]
        return reply, actions

    if intent == "bulk_help":
        reply = (
            "You can upload a CSV of multiple scenarios and download predictions with suggestions. "
            f"Use the bulk page here → {HELP_LINKS['bulk']}. The file should include columns like "
            "VISA_CLASS, JOB_TITLE, EMPLOYER_STATE, WORKSITE_STATE, WAGE_RATE_OF_PAY_FROM, etc."
        )
        actions = [{"label": "Open Bulk Upload", "href": HELP_LINKS["bulk"]}]
        return reply, actions

    if intent == "improve":
        reply = (
            "Here are practical steps that often improve approval likelihood:\n"
            "- " + "\n- ".join(IMPROVE_TIPS)
        )
        actions = [
            {"label": "Compare My Wage", "href": HELP_LINKS["wage"]},
            {"label": "Try Bulk What-ifs", "href": HELP_LINKS["bulk"]},
        ]
        return reply, actions

    if intent == "explain":
        reply = (
            "Your result blends model drivers (visualized on the result page) and domain rules "
            "(wage vs prevailing, full-time, compliance flags, duration). If your wage is below prevailing "
            "or the role isn’t full-time, that typically lowers probability. You can re-test several scenarios via bulk upload."
        )
        actions = [
            {"label": "Open Wage Comparator", "href": HELP_LINKS["wage"]},
            {"label": "Bulk What-ifs", "href": HELP_LINKS["bulk"]},
        ]
        return reply, actions

    if intent == "capability":
        reply = (
            "I can: (1) explain the model and drivers, (2) suggest ways to improve your chances, "
            "(3) compare wages by SOC/state, (4) run bulk CSV predictions, and (5) point you to the right page."
        )
        actions = [
            {"label": "Single Prediction", "href": HELP_LINKS["single"]},
            {"label": "Wage Comparator", "href": HELP_LINKS["wage"]},
            {"label": "Bulk Upload", "href": HELP_LINKS["bulk"]},
        ]
        return reply, actions

    if intent == "faq":
        return _run_faq(user_text), [{"label": "Home", "href": HELP_LINKS["single"]}]

    # Fallback
    reply = (
        "I can help with explanations, improvement tips, wage checks, and bulk predictions. "
        "Would you like to (a) compare wages, (b) upload a CSV, or (c) see what affects approval?"
    )
    actions = [
        {"label": "Wage Comparator", "href": HELP_LINKS["wage"]},
        {"label": "Bulk Upload", "href": HELP_LINKS["bulk"]},
        {"label": "What affects approval?", "payload": "what affects approval"},
    ]
    return reply, actions
