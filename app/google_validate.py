import requests
from bs4 import BeautifulSoup

def validate_field(job_title: str, state: str):
    try:
        q = f"{job_title} average H1B approval rate in {state} site:gov"
        url = f"https://www.google.com/search?q={q}"
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers)
        soup = BeautifulSoup(res.text, "html.parser")
        snippet = soup.select_one("div span")
        return snippet.text if snippet else "No relevant data found."
    except Exception:
        return "Could not verify online."
