import os
import pandas as pd
from datetime import datetime

REINFORCE_PATH = "../data/new_user_submissions.csv"

def log_submission(form_dict: dict, predicted_prob: float):
    os.makedirs(os.path.dirname(REINFORCE_PATH), exist_ok=True)
    row = form_dict.copy()
    row["predicted_prob"] = predicted_prob
    row["timestamp"] = datetime.utcnow().isoformat()
    df = pd.DataFrame([row])
    header = not os.path.exists(REINFORCE_PATH)
    df.to_csv(REINFORCE_PATH, mode="a", index=False, header=header)
