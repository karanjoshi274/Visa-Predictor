def validate_job_employer(job_title: str, employer_state: str):
    notes = []
    if not job_title or len(job_title) < 3:
        notes.append("Job title is short/unusual.")
    if employer_state and len(employer_state) != 2:
        notes.append("State should be 2-letter code (e.g., CA, NY).")
    if not notes:
        notes.append("Basic validations OK.")
    return " | ".join(notes)
