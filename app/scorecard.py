import math
from datetime import datetime

def _to_float(x, default=0.0):
    try:
        return float(str(x).strip())
    except Exception:
        return default

def compute_strength_score(form, derived=None, **kwargs):

    if derived is None:
        derived = {}

    f = {}
    if form:
        f.update(form)
    if kwargs:
        f.update(kwargs)


    def get(key_variants, default=None):
        for k in key_variants:
            if k in f and f[k] not in (None, ""):
                return f[k]
            # also try lowercase
            lk = k.lower()
            for fk in f:
                if fk.lower() == lk and f[fk] not in (None, ""):
                    return f[fk]
        return default

    offered_raw = get(['offered_wage', 'WAGE_RATE_OF_PAY_FROM', 'wage', 'WAGE_RATE_OF_PAY_FROM'], 0)
    prevailing_raw = get(['PREVAILING_WAGE', 'prevailing_wage', 'prev_wage'], 0)

    offered = _to_float(offered_raw, 0.0)
    prevailing = _to_float(prevailing_raw, 0.0)

    ft_raw = get(['FULL_TIME_POSITION', 'full_time_position', 'FULL_TIME'], 'N')
    ft = str(ft_raw).strip().upper() in ('Y', 'YES', 'TRUE', '1')

    h1b_dep = str(get(['H_1B_DEPENDENT', 'h1b_dependent'], 'N')).strip().upper() in ('Y','YES','TRUE','1')
    willful = str(get(['WILLFUL_VIOLATOR', 'willful_violator'], 'N')).strip().upper() in ('Y','YES','TRUE','1')
    agree_lc = str(get(['AGREE_TO_LC_STATEMENT', 'agree_to_lc_statement', 'AGREE_TO_LC'], 'Y')).strip().upper() in ('Y','YES','TRUE','1')

    duration_days = derived.get('DURATION_DAYS', None)
    if duration_days is None:
        begin = get(['BEGIN_DATE', 'begin_date'], None)
        end = get(['END_DATE', 'end_date'], None)
        try:
            if begin and end:
                b = datetime.fromisoformat(str(begin))
                e = datetime.fromisoformat(str(end))
                duration_days = max(0, (e - b).days)
        except Exception:
            duration_days = 0
    try:
        duration_days = int(duration_days or 0)
    except Exception:
        duration_days = 0

    wage_score = 0.0
    wage_note = ""
    if prevailing > 0:
        ratio = offered / prevailing if prevailing else 0.0

        if ratio < 0.8:
            wage_score = max(0.0, 40.0 * (ratio / 0.8))
        elif ratio < 1.0:
            wage_score = 40.0 + 30.0 * ((ratio - 0.8) / 0.2)
        else:
            # above prevailing increases score up to 100 for 1.5x or more
            wage_score = min(100.0, 70.0 + 30.0 * min( (ratio - 1.0) / 0.5, 1.0))
        wage_note = f"Offered/median ratio = {ratio:.2f} (median={prevailing:,.0f})"
    else:
        if offered >= 100000:
            wage_score = 70.0
        elif offered > 0:
            wage_score = 40.0
        else:
            wage_score = 0.0
        wage_note = "Prevailing wage not available"

    compliance_score = 100.0
    if willful:
        compliance_score -= 60.0
    if h1b_dep:
        compliance_score -= 20.0
    if not agree_lc:
        compliance_score -= 30.0
    compliance_score = max(0.0, min(100.0, compliance_score))

    stability_score = 50.0
    # boost for full-time
    stability_score += 20.0 if ft else -10.0
    # boost for longer durations
    if duration_days >= 365:
        stability_score += 20.0
    elif duration_days >= 180:
        stability_score += 10.0
    stability_score = max(0.0, min(100.0, stability_score))

    doc_score = 90.0

    missing = 0
    for key in ['JOB_TITLE','EMPLOYER_NAME','WORKSITE_STATE','WAGE_RATE_OF_PAY_FROM','BEGIN_DATE','END_DATE']:
        if not get([key]):
            missing += 1
    doc_score = max(0.0, 100.0 - missing*10.0)

    total_score = (wage_score * 0.35) + (compliance_score * 0.25) + (stability_score * 0.20) + (doc_score * 0.20)
    total_score = round(total_score, 1)

    return {
        'wage_score': round(wage_score, 1),
        'wage_note': wage_note,
        'compliance_score': round(compliance_score, 1),
        'stability_score': round(stability_score, 1),
        'documentation_score': round(doc_score, 1),
        'total_score': total_score
    }
